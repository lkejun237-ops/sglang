# SPDX-License-Identifier: Apache-2.0
"""Issue #195 encoder-only acceptance; service/media gates remain independent.

Run each mode/order in a fresh torchrun process on the same physical devices.
Thresholds are frozen in PERFORMANCE_ENCODER_PLAN.md, not CLI-tunable.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from frozen_source import verify_source
from sp1_reference import SP1Reference
from validate_incremental_codecs import digest, load_codecs
from validate_progressive_encoder import (
    candidate_future_checks,
    lifecycle_checks,
    run_stream,
    tensor_digest,
)
from validate_progressive_tiles import encoder_server_args, factory, indexed


def compare(actual, reference):
    same_format = actual.shape == reference.shape and actual.dtype == reference.dtype
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    bitwise = (
        same_format and finite and tensor_digest(actual) == tensor_digest(reference)
    )
    if same_format and finite:
        a, b = actual.double(), reference.double()
        delta = a - b
        maximum = delta.abs().max().item()
        relative = (delta.norm() / b.norm().clamp_min(1e-15)).item()
    else:
        maximum = relative = None
    return dict(
        passed=same_format and finite and maximum <= 0.25 and relative <= 0.005,
        bitwise=bitwise,
        same_format=same_format,
        finite=finite,
        max_abs=maximum,
        relative_l2=relative,
        actual_sha256=tensor_digest(actual),
        reference_sha256=tensor_digest(reference),
    )


def numerical_failure_blocks(result, *, required, report_only):
    """Diagnostic collection never permits nonfinite or malformed tensors."""
    return not result[required] and (
        not report_only or not result["finite"] or not result["same_format"]
    )


def verify_execution(timing, world, rank):
    for row in timing["rows"]:
        receipt = row["tile_execution"]
        expected = list(range(rank, 12, world)) if world > 1 else list(range(12))
        if (
            receipt["local_tile_ids"] != expected
            or receipt["tile_count"] != 12
            or receipt["requested_mode"] != "auto"
            or receipt["encoder_ranks"] != world
            or receipt["effective_mode"] != ("parallel" if world > 1 else "replicated")
        ):
            raise ValueError("Encoder did not execute the frozen tile partition")


def failure_checks(codecs, pixels, world, rank):
    results = []
    model = codecs.modules["video_vae"]
    for failure in (
        ["kernel"] if world == 1 else ["kernel", "state", "ended", "dispose", "dtype"]
    ):
        session = factory("auto")(codecs, capacity_frames=34)
        original = model.quant_conv.forward
        dtype = codecs.runtime["vae_config"]["encode_dtype"]
        backend = (
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        try:
            session.append(pixels[:, : 5 if failure == "ended" else 9], frame_start=0)
            session.pop()
            usage = session.state_usage()
            if rank == world - 1:
                if failure == "kernel":

                    def fail(*args):
                        raise RuntimeError("Injected encoder kernel failure")

                    model.quant_conv.forward = fail
                elif failure == "state":
                    session.reset()
                elif failure == "ended":
                    session.finish()
                elif failure == "dispose":
                    session.dispose()
                else:
                    codecs.runtime["vae_config"]["encode_dtype"] = "float32"
            try:
                session.pop()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Injected distributed failure was not rejected")
            if failure in ("kernel", "dtype") and session.state_usage() != usage:
                raise AssertionError("Failed encoder advanced cached state")
            if backend != (
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
                torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
            ):
                raise AssertionError("Backend state was not restored after failure")
            results.append(dict(failure=failure, passed=True))
        finally:
            model.quant_conv.forward = original
            codecs.runtime["vae_config"]["encode_dtype"] = dtype
            session.dispose()
            if session.state_usage().retained_bytes:
                raise AssertionError("Failed session retained cached storage")
    return results


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
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world not in (1, 2, 4):
        raise ValueError("Only SP1/SP2/SP4 experiments are defined")
    torch.cuda.set_device(local)
    if world > 1:
        init_distributed_environment(world_size=world, rank=rank, local_rank=local)
        initialize_model_parallel(sequence_parallel_degree=world, ulysses_degree=world)
    os.environ["MINIMAX_H3_PROGRESSIVE_ENCODER_PERFORMANCE_AUTO"] = "1"
    out = Path(args.output) / f"rank{rank}"
    out.mkdir(parents=True, exist_ok=False)
    if digest(args.rgb) != args.rgb_sha256:
        raise ValueError("Frozen source RGB hash mismatch")
    set_global_server_args(encoder_server_args(args.model))
    codecs, weights = load_codecs(Path(args.model))
    if codecs.runtime["vae_config"]["encode_mode"] != args.mode:
        raise ValueError("Model mode differs from requested experimental mode")
    if codecs.runtime["vae_config"]["encode_dtype"] != "bfloat16":
        raise ValueError("Experiment requires BF16")
    model = codecs.modules["video_vae"]
    model.stack_tiling = False
    model.encoder.to(dtype=torch.bfloat16)
    model.quant_conv.to(dtype=torch.bfloat16)
    raw = Path(args.rgb).read_bytes()
    if len(raw) < 1440 * 832 * 480 * 3 or len(raw) % (832 * 480 * 3):
        raise ValueError("Expected at least 1440 RGB24 832x480 frames")
    pixels = (
        torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        .reshape(-1, 480, 832, 3)
        .permute(3, 0, 1, 2)
    )
    prop = torch.cuda.get_device_properties(local)
    report = dict(
        status="running",
        args=vars(args),
        rank=rank,
        world=world,
        gpu_uuid=str(prop.uuid),
        gpu_name=prop.name,
        memory_bytes=prop.total_memory,
        compute_capability=[prop.major, prop.minor],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        weights_sha256=digest(weights),
        config_sha256=digest(Path(args.model) / "zing_config.json"),
        cases=[],
        timings=[],
        profiles=[],
        service_acceptance="pending",
        numerical_failures=[],
        numerical_acceptance="pending",
    )

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2))

    save()
    oracle = None
    deterministic_oracle = None
    if args.sp1_reference:
        provenance = dict(
            source_sha=args.source_sha,
            rgb_sha256=args.rgb_sha256,
            weights_sha256=report["weights_sha256"],
            config_sha256=report["config_sha256"],
            vae_config_sha256=digest(Path(args.model) / "video_vae/config.json"),
            script_sha256=digest(__file__),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            mode=args.mode,
            gpu_name=prop.name,
            compute_capability=[prop.major, prop.minor],
        )
        oracle = SP1Reference(args.sp1_reference, provenance, write=world == 1)
        if args.deterministic_reference:
            baseline_provenance = dict(
                provenance,
                mode="deterministic",
                config_sha256=args.deterministic_config_sha256,
            )
            deterministic_oracle = SP1Reference(
                args.deterministic_reference, baseline_provenance, write=False
            )
    try:
        # Cold and warmup include all three history states and a clip reset.
        _, report["cold"] = run_stream(
            codecs,
            pixels[:, :34],
            factory("auto"),
            [1],
            progressive=True,
            capture=False,
        )
        _, report["warmup"] = run_stream(
            codecs,
            pixels[:, :34],
            factory("auto"),
            [1],
            progressive=True,
            capture=False,
        )
        verify_execution(report["warmup"], world, rank)
        save()
        if args.phase == "correctness":
            for frames in [*range(1, 35), 120]:
                ref, ref_timing = run_stream(
                    codecs,
                    pixels[:, :frames],
                    factory("replicated"),
                    [1],
                    progressive=True,
                )
                ref_posterior = indexed(codecs, ref, ref_timing["rows"])
                sp1 = oracle.tensors(frames, ref, ref_posterior) if oracle else None
                baseline = (
                    deterministic_oracle.tensors(frames)
                    if deterministic_oracle
                    else None
                )
                packet_reference = None
                for pattern in ([1], [1, 2, 2], [17], [3, 11, 2, 19, 7]):
                    actual, timing = run_stream(
                        codecs,
                        pixels[:, :frames],
                        factory("auto"),
                        pattern,
                        progressive=True,
                    )
                    verify_execution(timing, world, rank)
                    if packet_reference is None:
                        packet_reference = tensor_digest(actual)
                    elif tensor_digest(actual) != packet_reference:
                        raise ValueError("Network packetization changed encoder bytes")
                    row = dict(
                        frames=frames,
                        pattern=pattern,
                        moments=compare(actual, ref),
                        posterior=compare(
                            indexed(codecs, actual, timing["rows"]), ref_posterior
                        ),
                        timeline=timing,
                    )
                    if sp1 is not None:
                        row["sp1_moments"] = compare(actual, sp1["moments"])
                        row["sp1_posterior"] = compare(
                            indexed(codecs, actual, timing["rows"]), sp1["posterior"]
                        )
                    if baseline is not None:
                        row["deterministic_moments"] = compare(
                            actual, baseline["moments"]
                        )
                        row["deterministic_posterior"] = compare(
                            indexed(codecs, actual, timing["rows"]),
                            baseline["posterior"],
                        )
                    if world > 1:
                        import torch.distributed as dist

                        hashes = [None] * world
                        dist.all_gather_object(hashes, tensor_digest(actual))
                        if len(set(hashes)) != 1:
                            raise ValueError(
                                "Gathered moments differ bitwise across ranks"
                            )
                    report["cases"].append(row)
                    save()
                    required = "bitwise" if args.mode == "deterministic" else "passed"
                    failures = [
                        key
                        for key in (
                            "moments",
                            "posterior",
                            "sp1_moments",
                            "sp1_posterior",
                            "deterministic_moments",
                            "deterministic_posterior",
                        )
                        if key in row and not row[key][required]
                    ]
                    if failures:
                        report["numerical_failures"].append(
                            dict(frames=frames, pattern=pattern, comparisons=failures)
                        )
                        save()
                        if len(report["numerical_failures"]) == 1:
                            torch.save(
                                dict(
                                    actual=actual,
                                    reference=ref,
                                    sp1=sp1,
                                    deterministic=baseline,
                                ),
                                out / "numerical-failure.pt",
                            )
                        if any(
                            numerical_failure_blocks(
                                row[key],
                                required=required,
                                report_only=args.report_numerical_only,
                            )
                            for key in failures
                        ):
                            raise ValueError("Frozen numerical gate failed")
            if oracle:
                oracle.finish()
            report["future"] = candidate_future_checks(
                codecs, pixels[:, :120], factory("auto"), out
            )
            if not all(x["passed"] for x in report["future"]):
                raise ValueError("Future frame invariance failed")
            report["lifecycle"] = lifecycle_checks(
                codecs,
                pixels[:, :120],
                factory("auto"),
                atol=0,
                rtol=0,
                output=out,
                reference_factory=factory("auto"),
            )
            report["failures"] = failure_checks(codecs, pixels, world, rank)
        else:
            for frames in (240, 1440):
                if args.phase == "timing":
                    for repeat in range(3):
                        _, timing = run_stream(
                            codecs,
                            pixels[:, :frames],
                            factory("auto"),
                            [1],
                            progressive=True,
                            capture=False,
                        )
                        verify_execution(timing, world, rank)
                        report["timings"].append(
                            dict(frames=frames, repeat=repeat, **timing)
                        )
                        save()
                else:
                    trace = out / f"encoder-{frames}.json"
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=True,
                    ) as prof:
                        _, timing = run_stream(
                            codecs,
                            pixels[:, :frames],
                            factory("auto"),
                            [1],
                            progressive=True,
                            capture=False,
                        )
                    prof.export_chrome_trace(str(trace))
                    verify_execution(timing, world, rank)
                    report["profiles"].append(dict(trace=trace.name, timeline=timing))
        if args.phase == "correctness":
            report["numerical_acceptance"] = (
                "failed" if report["numerical_failures"] else "passed"
            )
        report["status"] = "encoder_phase_complete_not_service_acceptance"
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
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--mode", choices=("deterministic", "performance"), required=True
    )
    parser.add_argument(
        "--phase", choices=("correctness", "timing", "profile"), required=True
    )
    parser.add_argument(
        "--report-numerical-only",
        action="store_true",
        help="Collect finite numerical differences without changing acceptance thresholds",
    )
    parser.add_argument("--sp1-reference")
    parser.add_argument("--deterministic-reference")
    parser.add_argument("--deterministic-config-sha256")
    args = parser.parse_args()
    if args.sp1_reference and args.phase != "correctness":
        parser.error("--sp1-reference requires correctness")
    if args.deterministic_reference and (
        args.mode != "performance"
        or not args.sp1_reference
        or not args.deterministic_config_sha256
    ):
        parser.error(
            "Deterministic reference requires performance correctness and frozen config hash"
        )
    run(args)
