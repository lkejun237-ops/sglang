# SPDX-License-Identifier: Apache-2.0
"""Real-weight dependency audit and adapter acceptance for issue #142.

Audit is deliberately NOT an incremental encoder implementation. No adapter is
selected implicitly; missing learned-kernel delivery cannot pass acceptance.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.progressive_encoder import (
    PROGRESSIVE_ENCODER_VERSION,
    expected_progressive_slices,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_codecs import (
    SourcePixelBuffer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.streaming_noise import (
    STREAMING_NOISE_VERSION,
)


def tensor_digest(value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def comparison(actual, expected, *, atol, rtol):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Encoder tensor shape or dtype differs from reference")
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    delta = a - b
    return dict(
        passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        and torch.allclose(a, b, atol=atol, rtol=rtol),
        max_abs=delta.abs().max().item(),
        rrmse=(delta.square().sum() / b.square().sum().clamp_min(1e-30)).sqrt().item(),
        actual_sha256=tensor_digest(actual),
        expected_sha256=tensor_digest(expected),
    )


def numerical_failure_blocks(result, *, report_only):
    # Diagnostic-only parity never permits NaN/Inf or relabels a mismatch.
    if not all(math.isfinite(result[key]) for key in ("max_abs", "rrmse")):
        return True
    return not result["passed"] and not report_only


def check_slices(rows, frames, *, progressive):
    actual = [(r["start"], r["end"], r["dependency"]) for r in rows]
    if progressive:
        expected = expected_progressive_slices(frames)
    else:
        expected = []
        for p in range(0, frames, 17):
            n = min(17, frames - p)
            start = p // 17 * 5
            expected.append((start, start + 1 + math.ceil((n - 1) / 4), p + n))
    if actual != expected:
        raise ValueError(f"Encoder slice/dependency mismatch: {actual} != {expected}")
    for row in rows:
        if row["dependency"] > row["accepted"]:
            raise ValueError("Encoder accessed future source frames")
    if progressive and (rows[0]["accepted"] >= 17 or rows[0]["eos"]):
        # Whole-clip transport cannot demonstrate first-frame readiness; caller
        # only enables this assertion for single-frame transport below.
        raise ValueError("First stable latent was not delivered before clip completion")


def run_stream(
    codecs, pixels, factory, pattern, *, paced=False, progressive=False, capture=True
):
    """Only reveal the current packet; record queue age against a fixed 24Hz clock."""
    stream = factory(codecs, capacity_frames=34)
    if (
        progressive
        and getattr(stream, "contract_version", None) != PROGRESSIVE_ENCODER_VERSION
    ):
        stream.dispose()
        raise ValueError("Unsupported progressive encoder contract")
    cursor = index = 0
    rows, values = [], []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    encode_seconds = 0.0

    def drain(eos=False):
        nonlocal encode_seconds
        while stream.ready:
            before = time.perf_counter()
            item = stream.pop()
            torch.cuda.synchronize()
            now = time.perf_counter()
            encode_seconds += now - before
            if item is None:
                raise ValueError("Ready encoder returned no result")
            elapsed = now - started
            rows.append(
                dict(
                    start=item.start,
                    end=item.end,
                    dependency=item.source_dependency_end,
                    accepted=cursor,
                    eos=eos,
                    seconds=elapsed,
                    pop_seconds=now - before,
                    queue_age_seconds=max(0.0, elapsed - (cursor - 1) / 24)
                    if paced
                    else None,
                    allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    state=vars(stream.state_usage()) if progressive else None,
                )
            )
            # Snapshot before the next call, detecting state/output aliasing in
            # the lifecycle checks separately. Copy is outside measured pop time.
            if capture:
                values.append(item.values.detach().cpu().clone())

    try:
        while cursor < pixels.shape[1]:
            count = min(pattern[index % len(pattern)], pixels.shape[1] - cursor)
            if paced:
                # Last pixel of a packet must have arrived before append.
                deadline = started + (cursor + count - 1) / 24
                time.sleep(max(0, deadline - time.perf_counter()))
            before = time.perf_counter()
            stream.append(
                pixels[:, cursor : cursor + count].clone(), frame_start=cursor
            )
            torch.cuda.synchronize()
            encode_seconds += time.perf_counter() - before
            cursor += count
            index += 1
            drain()
        stream.finish()
        drain(eos=True)
        elapsed = time.perf_counter() - started
        result = torch.cat(values, 2) if capture else None
        # Validate exact grouping even for whole-clip packets; early delivery is
        # independently checked on the single-pixel schedule.
        if not progressive:
            check_slices(rows, cursor, progressive=False)
        if progressive:
            actual = [(r["start"], r["end"], r["dependency"]) for r in rows]
            if actual != expected_progressive_slices(cursor):
                raise ValueError(
                    "Progressive output groups or dependency watermarks differ"
                )
            if any(r["dependency"] > r["accepted"] for r in rows):
                raise ValueError("Encoder accessed future source frames")
            if pattern == [1]:
                check_slices(rows, cursor, progressive=True)
                if any(r["accepted"] != r["dependency"] for r in rows if not r["eos"]):
                    raise ValueError(
                        "Progressive encoder missed a declared readiness watermark"
                    )
        intervals = [b["seconds"] - a["seconds"] for a, b in zip(rows, rows[1:])]
        ordered = sorted(intervals)
        return result, dict(
            first_latent_seconds=rows[0]["seconds"],
            output_interval_p50=statistics.median(intervals) if intervals else None,
            output_interval_p95=ordered[math.ceil(0.95 * len(ordered)) - 1]
            if ordered
            else None,
            output_interval_max=max(intervals) if intervals else None,
            pattern=pattern,
            paced=paced,
            rows=rows,
            seconds=elapsed,
            encode_host_seconds=encode_seconds,
            encode_host_fps=cursor / encode_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    finally:
        stream.dispose()


def lifecycle_checks(
    codecs, pixels, factory, *, atol, rtol, output=None, report_numerical_only=False
):
    """Exercise actual adapter caches; no synthetic encoder is installed here."""
    sources = [pixels[:, :34], 255 - pixels[:, :34]]
    references = [run_stream(codecs, x, SourcePixelBuffer, [17])[0] for x in sources]
    independent = [
        run_stream(codecs, x, factory, [1], progressive=True)[0] for x in sources
    ]
    sessions = [factory(codecs, capacity_frames=34) for _ in sources]
    held = [[], []]
    snapshots = [[], []]

    def consume(i):
        while sessions[i].ready:
            item = sessions[i].pop()
            if item is None:
                raise ValueError("Ready encoder returned no result")
            held[i].append(item.values)
            snapshots[i].append(item.values.detach().cpu().clone())

    def unchanged():
        for tensors, copies in zip(held, snapshots):
            for tensor, copy in zip(tensors, copies):
                if not comparison(tensor, copy, atol=0, rtol=0)["passed"]:
                    raise ValueError("Previously committed moments were overwritten")

    try:
        # Reject a gap without changing session cursors.
        for stream in sessions:
            try:
                stream.append(sources[0][:, :1], frame_start=1)
            except ValueError:
                pass
            else:
                raise ValueError("Encoder accepted a noncontiguous frame range")
            if stream.accepted_frames != 0 or stream.emitted_latents != 0:
                raise ValueError("Rejected append changed encoder cursors")
        for frame in range(34):
            for i, stream in enumerate(sessions):
                packet = sources[i][:, frame : frame + 1].clone()
                stream.append(packet, frame_start=frame)
                packet.zero_()  # Ownership must not remain with the caller.
                consume(i)
            unchanged()
        results = []
        for i, stream in enumerate(sessions):
            stream.finish()
            consume(i)
            if not comparison(
                torch.cat(snapshots[i], 2), independent[i], atol=0, rtol=0
            )["passed"]:
                raise ValueError("Interleaving changed candidate session output")
            result = comparison(
                torch.cat(snapshots[i], 2), references[i], atol=atol, rtol=rtol
            )
            results.append(result)
            if output is not None:
                torch.save(
                    dict(actual=torch.cat(snapshots[i], 2), expected=references[i]),
                    output / f"lifecycle-session-{i}.pt",
                )
                (output / "lifecycle-comparisons.json").write_text(
                    json.dumps(results, indent=2)
                )
            if numerical_failure_blocks(result, report_only=report_numerical_only):
                raise ValueError(
                    "Interleaved sessions differ from independent references"
                )
            stream.reset()
            if (
                stream.accepted_frames
                or stream.emitted_latents
                or stream.ready
                or stream.ended
            ):
                raise ValueError("Reset did not clear session state")
            # Reset again with pending input, then replay a complete clip.
            stream.append(sources[i][:, :2].clone(), frame_start=0)
            stream.reset()
            stream.append(sources[i][:, :17].clone(), frame_start=0)
            reset_values = []
            while stream.ready:
                reset_values.append(stream.pop().values.cpu().clone())
            replay = comparison(
                torch.cat(reset_values, 2),
                independent[i][:, :, :5],
                atol=0,
                rtol=0,
            )
            if not replay["passed"]:
                raise ValueError("Reset changed replayed moments")
            stream.dispose()
            stream.dispose()
            if stream.state_usage().retained_bytes != 0:
                raise ValueError("Dispose retained encoder-owned tensor storage")
            for action in (stream.reset, stream.finish, stream.pop):
                try:
                    action()
                except ValueError:
                    pass
                else:
                    raise ValueError("Disposed encoder accepted an operation")
            unchanged()
        return dict(
            interleaved=results, reset=True, output_ownership=True, dispose=True
        )
    finally:
        for stream in sessions:
            stream.dispose()


def candidate_future_checks(codecs, pixels, factory, output):
    """Perturb only unrevealed pixels; compare the real cached candidate itself."""
    reference, _ = run_stream(codecs, pixels[:, :17], factory, [1], progressive=True)
    torch.save(reference, output / "candidate-future-reference.pt")
    rows = []
    for count, tokens in ((1, 1), (9, 3)):
        changed = pixels[:, :17].clone()
        changed[:, count:] = 255 - changed[:, count:]
        actual, _ = run_stream(codecs, changed, factory, [1], progressive=True)
        torch.save(actual, output / f"candidate-future-after-{count}.pt")
        rows.append(
            dict(
                unchanged_pixels=count,
                **comparison(
                    actual[:, :, :tokens], reference[:, :, :tokens], atol=0, rtol=0
                ),
            )
        )
    return rows


def prefix_audit(codecs, pixels, output, atol, rtol):
    """Repeated-prefix diagnostic only: never registered as a stream factory."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        _visual_encode_policy,
    )

    model = codecs.modules["video_vae"]
    reference = codecs.encode_visual_moments(pixels[:, :17])
    device, dtype = reference.device, reference.dtype
    x = pixels[None, :, :17].to(device=device, dtype=dtype).div(255)
    mean = x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
    std = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
    x = (x - mean) / std
    rows = []
    for count, tokens in ((1, 1), (9, 3), (17, 5)):
        with _visual_encode_policy(codecs.runtime["vae_config"]["encode_mode"]):
            actual = model._adaptive_encode(x[:, :, :count])
        expected = reference[:, :, :tokens]
        rows.append(
            dict(
                pixels=count,
                tokens=tokens,
                **comparison(actual, expected, atol=atol, rtol=rtol),
            )
        )
        torch.save(actual.cpu(), output / f"audit-prefix-{count}.pt")
    future_checks = []
    for count, tokens in ((1, 1), (9, 3)):
        changed = pixels[:, :17].clone()
        changed[:, count:] = 255 - changed[:, count:]
        actual = codecs.encode_visual_moments(changed)[:, :, :tokens]
        future_checks.append(
            dict(
                unchanged_pixels=count,
                **comparison(actual, reference[:, :, :tokens], atol=atol, rtol=rtol),
            )
        )
        torch.save(actual.cpu(), output / f"audit-future-after-{count}.pt")
    torch.save(reference.cpu(), output / "audit-clip17.pt")
    return dict(
        scope="repeated-prefix dependency probe; not incremental/cache/performance acceptance",
        comparisons=rows,
        future_perturbations=future_checks,
    )


@torch.inference_mode()
def run(args):
    from validate_incremental_codecs import digest, load_codecs
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world not in (1, 2):
        raise ValueError("Progressive encoder acceptance supports SP1 or SP2 only")
    torch.cuda.set_device(local_rank)
    if world > 1:
        from sglang.multimodal_gen.runtime.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank)
        initialize_model_parallel(sequence_parallel_degree=world, ulysses_degree=world)
    output = Path(args.output)
    if world > 1:
        output = output / f"rank{rank}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        vars(args), input_sha256=digest(args.rgb), noise_version=STREAMING_NOISE_VERSION
    )
    manifest["script_sha256"] = digest(__file__)
    (output / "frozen.json").write_text(json.dumps(manifest, indent=2))
    if args.mode == "validate" and not args.adapter:
        raise ValueError(
            "Real progressive encoder adapter is required; no fallback is allowed"
        )
    set_global_server_args(
        ServerArgs.from_kwargs(
            model_path=args.model,
            num_gpus=1,
            attention_backend="torch_sdpa",
            warmup_mode="off",
        )
    )
    codecs, weights = load_codecs(Path(args.model))
    # Prepare shared weights once, before any live session. Do not rely on a
    # baseline encode call to change dtype as an incidental side effect.
    model = codecs.modules["video_vae"]
    encode_dtype = getattr(torch, codecs.runtime["vae_config"]["encode_dtype"])
    model.encoder.to(dtype=encode_dtype)
    model.quant_conv.to(dtype=encode_dtype)
    manifest.update(
        weights_sha256=digest(weights),
        config_sha256=digest(Path(args.model) / "zing_config.json"),
        vae_config_sha256=digest(Path(args.model) / "video_vae/config.json"),
        runtime=codecs.runtime,
        prepared_encoder_dtype=str(next(model.encoder.parameters()).dtype),
        prepared_quant_conv_dtype=str(next(model.quant_conv.parameters()).dtype),
    )
    prop = torch.cuda.get_device_properties(local_rank)
    manifest["hardware"] = dict(
        name=prop.name,
        memory_mib=prop.total_memory // 2**20,
        cc=[prop.major, prop.minor],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        sp=world,
        rank=rank,
    )
    (output / "frozen.json").write_text(json.dumps(manifest, indent=2))
    raw = Path(args.rgb).read_bytes()
    if len(raw) != 120 * 480 * 832 * 3:
        raise ValueError("Acceptance input must be 120 frames of 832x480 RGB24")
    pixels = (
        torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        .reshape(120, 480, 832, 3)
        .permute(3, 0, 1, 2)
        .contiguous()
    )
    report = dict(
        status="incomplete",
        scope=args.mode,
        cases=[],
        timings=[],
        missing=[
            "DiT consumption threshold",
            "end-to-end quality and FPS",
            "independent trace",
            "candidate future-frame perturbation",
            "long-stream bounded cache and linear compute",
            "reset/cancel/interleaved sessions",
        ],
    )

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps(
                dict(
                    rank=rank,
                    status=report["status"],
                    cases=len(report["cases"]),
                    timings=len(report["timings"]),
                )
            ),
            flush=True,
        )

    save()
    try:
        if args.mode == "audit":
            report["audit"] = prefix_audit(codecs, pixels, output, args.atol, args.rtol)
            report["missing"].append("progressive kernel delivery and integration")
        else:
            module, name = args.adapter.split(":")
            adapter_module = importlib.import_module(module)
            factory = getattr(adapter_module, name)
            manifest["adapter_file_sha256"] = digest(adapter_module.__file__)
            (output / "frozen.json").write_text(json.dumps(manifest, indent=2))
            report["candidate_future"] = candidate_future_checks(
                codecs, pixels, factory, output
            )
            save()
            if not all(row["passed"] for row in report["candidate_future"]):
                raise ValueError("Progressive candidate depends on future pixels")
            report["missing"].remove("candidate future-frame perturbation")
            report["lifecycle"] = lifecycle_checks(
                codecs,
                pixels,
                factory,
                atol=args.atol,
                rtol=args.rtol,
                output=output,
                report_numerical_only=args.report_numerical_only,
            )
            report["missing"].remove("reset/cancel/interleaved sessions")
            save()
            rng = random.Random(142)
            patterns = [[1], [1, 2, 2], [17], [rng.randint(1, 23) for _ in range(19)]]
            for count in [*range(1, 35), 120]:
                source = pixels[:, :count]
                expected, _ = run_stream(codecs, source, SourcePixelBuffer, [17])
                expected_sampled = codecs.sample_streaming_visual_moments(
                    expected.cuda(),
                    latent_start=0,
                    noise_version=STREAMING_NOISE_VERSION,
                )
                for pattern in patterns:
                    actual, timing = run_stream(
                        codecs, source, factory, pattern, progressive=True
                    )
                    sampled = torch.cat(
                        [
                            codecs.sample_streaming_visual_moments(
                                actual[:, :, r["start"] : r["end"]].cuda(),
                                latent_start=r["start"],
                                noise_version=STREAMING_NOISE_VERSION,
                            )
                            for r in timing["rows"]
                        ],
                        2,
                    )
                    case = dict(
                        frames=count,
                        pattern=pattern,
                        moments=comparison(
                            actual, expected, atol=args.atol, rtol=args.rtol
                        ),
                        sampled=comparison(
                            sampled, expected_sampled, atol=args.atol, rtol=args.rtol
                        ),
                        timeline=timing,
                    )
                    report["cases"].append(case)
                    save()
                    if not case["moments"]["passed"] or not case["sampled"]["passed"]:
                        torch.save(
                            dict(actual=actual, expected=expected),
                            output / "failed-tensors.pt",
                        )
                        if any(
                            numerical_failure_blocks(
                                case[key], report_only=args.report_numerical_only
                            )
                            for key in ("moments", "sampled")
                        ):
                            raise ValueError("Frozen numerical gate failed")
        if args.dependency_only:
            report["status"] = "dependency_probe_complete_not_acceptance"
            save()
            return
        factories = [("clip17", SourcePixelBuffer)]
        if args.mode == "validate":
            factories.append(("progressive", factory))
        for label, impl in factories:
            run_stream(
                codecs,
                pixels,
                impl,
                [1],
                progressive=label == "progressive",
                capture=False,
            )
            for paced in (False, True):
                for repeat in range(3):
                    _, timing = run_stream(
                        codecs,
                        pixels,
                        impl,
                        [1],
                        paced=paced,
                        progressive=label == "progressive",
                        capture=False,
                    )
                    report["timings"].append(dict(label=label, repeat=repeat, **timing))
                    save()
        report["status"] = (
            "audit_complete_not_acceptance"
            if args.mode == "audit"
            else "partial_validation_not_acceptance"
        )
        save()
    except Exception as error:
        report.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--rgb", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument(
        "--report-numerical-only",
        action="store_true",
        help="Record baseline numerical differences without blocking functional checks; candidate isolation remains exact",
    )
    p.add_argument("--mode", choices=["audit", "validate"], required=True)
    p.add_argument(
        "--dependency-only",
        action="store_true",
        help="Audit dependency without repeating the timing baseline",
    )
    p.add_argument("--adapter", help="Real kernel module:factory; never a mock")
    p.add_argument("--atol", type=float, default=0)
    p.add_argument("--rtol", type=float, default=0)
    args = p.parse_args()
    if args.dependency_only and args.mode != "audit":
        p.error("Dependency-only mode requires audit")
    if args.mode == "validate" and not args.adapter:
        p.error("Real progressive encoder adapter is required; no fallback is allowed")
    if min(args.atol, args.rtol) < 0 or not all(
        math.isfinite(x) for x in (args.atol, args.rtol)
    ):
        p.error("Tolerances must be finite and nonnegative")
    run(args)
