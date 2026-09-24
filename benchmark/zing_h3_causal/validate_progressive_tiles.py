# SPDX-License-Identifier: Apache-2.0
"""Issue #177 real-weight tile comparison. No tolerance/report-only escape hatch.

Run under torchrun separately for SP1/2/4/8 and each stack policy. Outputs are
private artifacts: raw moments and source-derived tensors must not be published.
This encoder-only tool never certifies the required service A/B or default gate.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from frozen_source import verify_source
from sp1_reference import PROVENANCE_KEYS, SP1Reference
from validate_incremental_codecs import digest, load_codecs
from validate_progressive_encoder import (
    candidate_future_checks,
    lifecycle_checks,
    run_stream,
    tensor_digest,
)

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
    create_session,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_codecs import (
    SourcePixelBuffer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_noise import (
    STREAMING_NOISE_VERSION,
)


def factory(mode):
    def make(codecs, *, capacity_frames):
        # Session construction snapshots the mode; the benchmark is single threaded.
        if mode == "auto":
            os.environ.pop("MINIMAX_H3_PROGRESSIVE_ENCODER_TILING", None)
        else:
            os.environ["MINIMAX_H3_PROGRESSIVE_ENCODER_TILING"] = mode
        return create_session(codecs, capacity_frames=capacity_frames)

    return make


def exact(actual, expected):
    same_format = actual.shape == expected.shape and actual.dtype == expected.dtype
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    actual_sha, expected_sha = tensor_digest(actual), tensor_digest(expected)
    return dict(
        passed=same_format and finite and actual_sha == expected_sha,
        same_format=same_format,
        finite=finite,
        actual_sha256=actual_sha,
        expected_sha256=expected_sha,
    )


def indexed(codecs, moments, rows):
    return torch.cat(
        [
            codecs.sample_streaming_visual_moments(
                moments[:, :, r["start"] : r["end"]].cuda(),
                latent_start=r["start"],
                noise_version=STREAMING_NOISE_VERSION,
            ).cpu()
            for r in rows
        ],
        2,
    )


def encoder_server_args(model_path):
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3 import (
        ZingH3CausalPipelineConfig,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    # This probe loads the source VAE, never a video decoder. Runtime's default
    # TAEH3 backend requires a checkpoint unrelated to encoder validation.
    return ServerArgs.from_kwargs(
        model_path=model_path,
        pipeline_config=ZingH3CausalPipelineConfig(video_decoder_backend="native"),
        num_gpus=1,
        attention_backend="torch_sdpa",
        warmup_mode="off",
    )


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
    world, rank, local_rank = (
        int(os.environ.get(k, default))
        for k, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"))
    )
    if world not in (1, 2, 4, 8):
        raise ValueError("Expected SP1/2/4/8")
    torch.cuda.set_device(local_rank)
    if world > 1:
        init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank)
        initialize_model_parallel(sequence_parallel_degree=world, ulysses_degree=world)
    output = Path(args.output) / f"rank{rank}"
    output.mkdir(parents=True, exist_ok=False)
    if digest(args.rgb) != args.rgb_sha256:
        raise ValueError("Source RGB SHA256 differs from the frozen input")
    set_global_server_args(encoder_server_args(args.model))
    codecs, weights = load_codecs(Path(args.model))
    model = codecs.modules["video_vae"]
    model.stack_tiling = args.stack_tiling == "on"
    dtype = getattr(torch, codecs.runtime["vae_config"]["encode_dtype"])
    model.encoder.to(dtype=dtype)
    model.quant_conv.to(dtype=dtype)
    prop = torch.cuda.get_device_properties(local_rank)
    frozen = dict(
        vars(args),
        rank=rank,
        world=world,
        runtime=codecs.runtime,
        weights_sha256=digest(weights),
        config_sha256=digest(Path(args.model) / "zing_config.json"),
        vae_config_sha256=digest(Path(args.model) / "video_vae/config.json"),
        script_sha256=digest(__file__),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu_name=prop.name,
        gpu_uuid=str(getattr(prop, "uuid", "unavailable")),
        memory_bytes=prop.total_memory,
        compute_capability=[prop.major, prop.minor],
        nccl=torch.cuda.nccl.version(),
        tile_size=model.tile_size,
        tile_overlap_min=model.tile_overlap_min,
        encoder_tiling=model.encoder_tiling,
        tile_count=len(model.split_tiles(480)[0]) * len(model.split_tiles(832)[0]),
    )
    (output / "frozen.json").write_text(json.dumps(frozen, indent=2))
    raw = Path(args.rgb).read_bytes()
    frame_bytes = 480 * 832 * 3
    if len(raw) % frame_bytes or len(raw) < 1440 * frame_bytes:
        raise ValueError("Provide at least 1440 real source frames, 832x480 RGB24")
    pixels = (
        torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        .reshape(-1, 480, 832, 3)
        .permute(3, 0, 1, 2)
    )
    report = dict(
        status="incomplete",
        cases=[],
        timings=[],
        profiles=[],
        service_ab="not_run",
        candidate_mode=args.parallel_mode,
    )

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2))

    sp1_reference = (
        SP1Reference(
            args.sp1_reference,
            {key: frozen[key] for key in PROVENANCE_KEYS},
            write=world == 1,
        )
        if args.sp1_reference
        else None
    )
    report["sp1_reference"] = (
        {"path": args.sp1_reference, "role": "writer" if world == 1 else "reader"}
        if sp1_reference is not None
        else None
    )
    save()
    try:
        if args.phase in ("correctness", "all"):
            rng = random.Random(177)
            patterns = [[1], [1, 2, 2], [17], [rng.randint(1, 23) for _ in range(19)]]
            for frames in [*range(1, 35), 120]:
                source = pixels[:, :frames]
                expected, reference = run_stream(
                    codecs, source, factory("replicated"), [1], progressive=True
                )
                expected_posterior = indexed(codecs, expected, reference["rows"])
                sp1 = (
                    sp1_reference.tensors(frames, expected, expected_posterior)
                    if sp1_reference is not None
                    else None
                )
                for pattern in patterns:
                    actual, timing = run_stream(
                        codecs,
                        source,
                        factory(args.parallel_mode),
                        pattern,
                        progressive=True,
                    )
                    row = dict(
                        frames=frames,
                        pattern=pattern,
                        moments=exact(actual, expected),
                        posterior=exact(
                            indexed(codecs, actual, timing["rows"]), expected_posterior
                        ),
                        timeline=timing,
                    )
                    if sp1 is not None:
                        row["sp1_moments"] = exact(actual, sp1["moments"])
                        row["sp1_posterior"] = exact(
                            indexed(codecs, actual, timing["rows"]), sp1["posterior"]
                        )
                    report["cases"].append(row)
                    save()
                    if not all(
                        row[key]["passed"]
                        for key in (
                            "moments",
                            "posterior",
                            "sp1_moments",
                            "sp1_posterior",
                        )
                        if key in row
                    ):
                        torch.save(
                            dict(actual=actual, expected=expected),
                            output / "exact-failure.pt",
                        )
                        raise ValueError("Exact replicated/tile-parallel gate failed")
            report["future"] = candidate_future_checks(
                codecs, pixels[:, :120], factory(args.parallel_mode), output
            )
            if not all(row["passed"] for row in report["future"]):
                raise ValueError("Future source dependency gate failed")
            report["lifecycle"] = lifecycle_checks(
                codecs,
                pixels[:, :120],
                factory(args.parallel_mode),
                atol=0,
                rtol=0,
                output=output,
                report_numerical_only=False,
                reference_factory=factory("replicated"),
            )
            if sp1_reference is not None:
                sp1_reference.finish()
            save()
        if args.phase in ("timing", "profile", "all"):
            for frames in (240, 1440):
                source = pixels[:, :frames]
                for label, impl in (
                    ("clip17", SourcePixelBuffer),
                    ("replicated", factory("replicated")),
                    ("parallel", factory(args.parallel_mode)),
                ):
                    progressive = label != "clip17"
                    for paced in (
                        (False, True)
                        if args.delivery == "both"
                        else (args.delivery == "paced",)
                    ):
                        # Separate warmup per delivery mode; never mix profiler samples.
                        run_stream(
                            codecs,
                            source,
                            impl,
                            [1],
                            progressive=progressive,
                            paced=paced,
                            capture=False,
                        )
                        if args.phase in ("timing", "all"):
                            for repeat in range(3):
                                _, timing = run_stream(
                                    codecs,
                                    source,
                                    impl,
                                    [1],
                                    progressive=progressive,
                                    paced=paced,
                                    capture=False,
                                )
                                report["timings"].append(
                                    dict(
                                        label=label,
                                        frames=frames,
                                        repeat=repeat,
                                        **timing,
                                    )
                                )
                                save()
                        if args.phase in ("profile", "all"):
                            trace = output / f"{label}-{frames}-paced{int(paced)}.json"
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
                                    impl,
                                    [1],
                                    progressive=progressive,
                                    paced=paced,
                                    capture=False,
                                )
                            prof.export_chrome_trace(str(trace))
                            report["profiles"].append(trace.name)
                            save()
        report["status"] = "encoder_phase_complete_not_service_acceptance"
        save()
    except Exception as error:
        report.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--rgb-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument(
        "--source-manifest", help="Frozen archive file hashes (CPU-prepared workers)"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sp1-reference",
        help="Private tensor oracle: SP1 writes; subsequent SP2/4/8 runs compare to it",
    )
    parser.add_argument(
        "--delivery", choices=("both", "offline", "paced"), default="both"
    )
    parser.add_argument("--stack-tiling", choices=("on", "off"), required=True)
    parser.add_argument(
        "--parallel-mode",
        choices=("parallel", "auto"),
        default="parallel",
        help="Use auto to validate the default with the tiling environment variable unset",
    )
    parser.add_argument(
        "--phase", choices=("correctness", "timing", "profile", "all"), default="all"
    )
    args = parser.parse_args()
    if args.sp1_reference and args.phase not in ("correctness", "all"):
        parser.error("--sp1-reference requires the correctness phase")
    run(args)
