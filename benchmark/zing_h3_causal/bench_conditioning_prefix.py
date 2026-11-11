# SPDX-License-Identifier: Apache-2.0
"""Compare frozen full Qwen encoding with task-local incremental encoding.

Inputs: JSON list of {name, prompt, updates, media:[{type,path,fps}]}.
Media paths are RGB images (a video entry represents its retained first frame).
No DiT/VAE execution or end-to-end performance claim is made by this tool.
"""

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image


def metrics(reference, actual):
    # Long flattened FP32 reductions can produce cosine > 1 even for identical
    # features. Accumulate validation in FP64, outside the measured encoding.
    a, b = reference.double(), actual.double()
    if a.numel() == 0:
        return {"rows": 0}
    diff = a - b
    return {
        "rows": len(a),
        "finite": bool(torch.isfinite(b).all()),
        "equal": bool(torch.equal(reference, actual)),
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "relative_l2": (diff.norm() / a.norm().clamp_min(1e-12)).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0
        ).item(),
        "min_row_cosine": torch.nn.functional.cosine_similarity(a, b, dim=1)
        .min()
        .item(),
    }


def measure(function):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    begin = time.perf_counter()
    start.record()
    result = function()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - begin) * 1000
    receipt = {
        "wall_ms": wall_ms,
        "cuda_stream_ms": start.elapsed_time(end),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "resident_allocated_after": torch.cuda.memory_allocated(),
    }
    return result.cpu(), receipt


def main(args):
    from sglang.multimodal_gen.runtime.conditioning.encoder import encode_condition
    from sglang.multimodal_gen.runtime.conditioning.inputs import content_identity
    from sglang.multimodal_gen.runtime.conditioning.native import (
        NativeConditionEncoder,
        build_profile,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.models.encoders.qwen_prefix import (
        QwenPrefixCache,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    hardware = dict(
        gpu=torch.cuda.get_device_name(),
        cc=".".join(map(str, torch.cuda.get_device_capability())),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        cuda=str(torch.version.cuda),
        memory_bytes=torch.cuda.get_device_properties(0).total_memory,
    )
    environment = dict(
        encoder_tp=1,
        gpu=hardware["gpu"],
        cc=hardware["cc"],
        precision="bf16",
        backend=args.backend,
        software={k: hardware[k] for k in ("torch", "transformers", "cuda")},
    )
    profile = build_profile(args.model, environment)
    native = NativeConditionEncoder(args.model, profile, max_rows=16384, gpu_id=0)
    encoder = native.pipeline.modules["text_encoder"]
    # Execute the pre-change method against the exact same loaded weights.
    source = args.baseline_source.read_text()
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3Qwen3VLEncoder"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "encode_ids"
    )
    namespace = {"torch": torch, "Any": Any}
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(args.baseline_source),
            "exec",
        ),
        namespace,
    )
    baseline = MethodType(namespace["encode_ids"], encoder)
    results, checks, native_samples = [], [], []
    metadata = dict(
        hardware=hardware,
        loader=native.receipt,
        profile=profile.document,
        baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        repeats=args.repeats,
        validation_dtype="float64",
        tolerance=dict(relative_l2=0.01, cosine=0.9999, min_row_cosine=0.999),
        scope="Qwen component only; no DiT, VAE, network or video quality acceptance",
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def save():
        (args.output / "samples.json").write_text(json.dumps(results, indent=2))
        (args.output / "checks.json").write_text(json.dumps(checks, indent=2))
        (args.output / "native_samples.json").write_text(
            json.dumps(native_samples, indent=2)
        )

    with (
        torch.inference_mode(),
        set_forward_context(current_timestep=0, attn_metadata=None),
    ):
        for case in json.loads(args.cases.read_text()):
            media = []
            for item in case["media"]:
                pixels = np.array(Image.open(item["path"]).convert("RGB"))
                media.append(
                    dict(
                        type=item["type"],
                        fps=item.get("fps", 24.0),
                        frames=torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(1),
                    )
                )
            identity = content_identity("media-only", media, profile.profile_id)
            prepared = []
            for prompt in [case["prompt"], *case["updates"]]:
                captured = {}

                def capture(ids, **features):
                    captured.update(ids=ids.clone(), features=features)
                    return torch.empty((ids.numel(), 1))

                modules = dict(
                    native.pipeline.modules,
                    text_encoder=SimpleNamespace(encode_ids=capture),
                )
                started = time.perf_counter()
                output = encode_condition(
                    modules, native.pipeline.zing_runtime, prompt, media, 16384
                )
                captured.update(
                    preprocess_ms=(time.perf_counter() - started) * 1000,
                    row_role=output["row_role"],
                )
                prepared.append(captured)
            cache = QwenPrefixCache(args.cache_bytes)

            def full(item):
                return baseline(item["ids"], **item["features"])

            def incremental(item, media_key=identity):
                return encoder.encode_ids(
                    item["ids"],
                    **item["features"],
                    prefix_cache=cache,
                    media_identity=media_key,
                )

            # Warm both shapes and establish cold-cache parity before timings.
            reference, _ = measure(lambda: full(prepared[0]))
            cold, _ = measure(lambda: incremental(prepared[0]))
            checks.append(
                dict(case=case["name"], kind="cold", **metrics(reference, cold))
            )
            for item in prepared[1:]:
                measure(lambda: full(item))
                measure(lambda: incremental(item))
            cache.clear()
            _, full_without_cache = measure(lambda: full(prepared[0]))
            checks.append(
                dict(
                    case=case["name"],
                    kind="full_without_cache_memory",
                    timing=full_without_cache,
                )
            )
            for repetition in range(args.repeats):
                measure(lambda: incremental(prepared[0]))
                previous = cache.context.cpu().clone()
                for index, item in enumerate(prepared[1:]):
                    # Full and incremental alternate in a single resident model;
                    # the baseline does not modify this task's separate KV.
                    ref, base_time = measure(lambda: full(item))
                    actual, inc_time = measure(lambda: incremental(item))
                    prefix = cache.reused_rows
                    parity = metrics(ref, actual)
                    parity["reused_prefix_bitwise"] = bool(
                        torch.equal(previous[:prefix], actual[:prefix])
                    )
                    row = dict(
                        case=case["name"],
                        repetition=repetition,
                        update=index,
                        total_rows=item["ids"].numel(),
                        reused_rows=prefix,
                        encoded_rows=cache.encoded_rows,
                        media_identity=identity,
                        input_shapes=[list(x["frames"].shape) for x in media],
                        full=base_time,
                        incremental=inc_time,
                        parity=parity,
                        suffix=metrics(ref[prefix:], actual[prefix:]),
                        boundary=metrics(
                            ref[max(0, prefix - 1) : prefix + 2],
                            actual[max(0, prefix - 1) : prefix + 2],
                        ),
                        preprocess_ms=item["preprocess_ms"],
                    )
                    results.append(row)
                    previous = actual
                    save()
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "case",
                                    "repetition",
                                    "update",
                                    "total_rows",
                                    "reused_rows",
                                    "parity",
                                )
                            }
                        ),
                        flush=True,
                    )
            # Exact-repeat shortcut, shortening, media invalidation and cached
            # output isolation must be checked independently from timing wins.
            last = prepared[-1]
            ref, _ = measure(lambda: full(last))
            same, _ = measure(lambda: incremental(last))
            checks.append(
                dict(
                    case=case["name"],
                    kind="repeat",
                    reused_rows=cache.reused_rows,
                    **metrics(ref, same),
                )
            )
            changed, _ = measure(lambda: incremental(last, "changed-media-identity"))
            checks.append(
                dict(
                    case=case["name"],
                    kind="media_invalidated",
                    reused_rows=cache.reused_rows,
                    **metrics(ref, changed),
                )
            )
            cache.clear()
            # Exercise the actual service encoder boundary, including hashing,
            # processor/tokenizer, H2D, full feature assembly and D2H. HTTP,
            # admission queueing and the downstream DiT remain outside timing.
            task = SimpleNamespace(cache=QwenPrefixCache(args.cache_bytes))
            candidate_method = encoder.encode_ids
            for repetition in range(3):
                native(SimpleNamespace(prompt=case["prompt"], media=media, task=task))
                for index, prompt in enumerate(case["updates"]):
                    full_result, inc_result = {}, {}

                    def native_full():
                        encoder.encode_ids = baseline
                        try:
                            full_result.update(
                                native(SimpleNamespace(prompt=prompt, media=media))
                            )
                            return full_result["context"]
                        finally:
                            encoder.encode_ids = candidate_method

                    def native_incremental():
                        inc_result.update(
                            native(
                                SimpleNamespace(prompt=prompt, media=media, task=task)
                            )
                        )
                        return inc_result["context"]

                    ref, base_time = measure(native_full)
                    actual, inc_time = measure(native_incremental)
                    parity = metrics(ref, actual)
                    parity["row_role_equal"] = bool(
                        torch.equal(full_result["row_role"], inc_result["row_role"])
                    )
                    native_samples.append(
                        dict(
                            case=case["name"],
                            repetition=repetition,
                            update=index,
                            full=base_time,
                            incremental=inc_time,
                            parity=parity,
                            reused_rows=task.cache.reused_rows,
                            suffix=metrics(
                                ref[task.cache.reused_rows :],
                                actual[task.cache.reused_rows :],
                            ),
                        )
                    )
                    save()
            task.cache.clear()
            save()
    passed = all(
        r["parity"]["finite"]
        and r["parity"]["relative_l2"] <= 0.01
        and r["parity"]["cosine"] >= 0.9999
        and r["parity"]["cosine"] <= 1.0 + 1e-12
        and r["parity"]["min_row_cosine"] >= 0.999
        and r["parity"]["reused_prefix_bitwise"]
        and (r["suffix"]["rows"] == 0 or r["suffix"]["relative_l2"] <= 0.01)
        for r in results
    )
    passed &= all(
        r["finite"] and r["relative_l2"] <= 0.01 for r in checks if "finite" in r
    )
    passed &= all(
        r["parity"]["finite"]
        and r["parity"]["row_role_equal"]
        and r["parity"]["relative_l2"] <= 0.01
        and 0.9999 <= r["parity"]["cosine"] <= 1.0 + 1e-12
        and r["parity"]["min_row_cosine"] >= 0.999
        and (r["suffix"]["rows"] == 0 or r["suffix"]["relative_l2"] <= 0.01)
        for r in native_samples
    )
    summary = {}
    for name in sorted({r["case"] for r in results}):
        selected = [r for r in results if r["case"] == name]
        values = {}
        for mode in ("full", "incremental"):
            times = [r[mode]["wall_ms"] for r in selected]
            values[mode] = dict(
                n=len(times),
                median_ms=float(np.median(times)),
                p95_ms=float(np.percentile(times, 95)),
                peak_allocated=max(r[mode]["peak_allocated"] for r in selected),
            )
        values["speedup"] = (
            values["full"]["median_ms"] / values["incremental"]["median_ms"]
        )
        values["max_relative_l2"] = max(r["parity"]["relative_l2"] for r in selected)
        values["max_suffix_relative_l2"] = max(
            r["suffix"].get("relative_l2", 0) for r in selected
        )
        values["native_boundary"] = {}
        for mode in ("full", "incremental"):
            times = [r[mode]["wall_ms"] for r in native_samples if r["case"] == name]
            values["native_boundary"][mode] = dict(
                n=len(times),
                median_ms=float(np.median(times)),
                p95_ms=float(np.percentile(times, 95)),
            )
        summary[name] = values
    (args.output / "RESULT.json").write_text(
        json.dumps(dict(passed=passed, summary=summary), indent=2)
    )
    print(json.dumps(dict(passed=passed, summary=summary)), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if not passed:
        raise SystemExit("Incremental feature parity gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="torch_sdpa", choices=("torch_sdpa", "fa"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cache-bytes", type=int, default=4 << 30)
    main(parser.parse_args())
