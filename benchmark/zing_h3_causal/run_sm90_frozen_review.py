# SPDX-License-Identifier: Apache-2.0
"""Sequential Encoder -> frozen DiT -> paired Decoder review on one GPU.

This is an offline component experiment, never an E2E serving benchmark.
Uses native preparation, progressive source VAE, queued DiT and Lynn streams.
Trusted local torch captures contain pickle; do not accept external captures.
"""

import argparse
import asyncio
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace

import torch
from bench_streaming_source import prepare_case
from benchmark_artifacts import MediaCapture

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.component_capture import (
    map_tensors,
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def save(path, value):
    torch.save(map_tensors(value, lambda x: x.detach().cpu().clone()), path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def server_args(args):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        initialize_runtime_imports,
    )

    ServerArgs, set_global = initialize_runtime_imports()
    server = ServerArgs.from_kwargs(
        model_path=str(args.model),
        pipeline_class_name="ZingH3CausalPipeline",
        num_gpus=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        attention_backend="torch_sdpa",
        video_decoder_backend="native",
        performance_mode="speed",
        warmup_mode="off",
        text_encoder_cpu_offload=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=False,
        use_fsdp_inference=False,
        mem_fraction_static=None,
        zing_solver="few_step",
        zing_denoising_step_list=[1000, 500],
        zing_initial_block_count=3,
        zing_initial_denoising_step_list=[1000, 750, 500, 250],
    )
    server.pipeline_config.zing_adaln_mode = args.adaln_mode
    server.pipeline_config.zing_adaln_table_path = (
        str(args.adaln_table) if args.adaln_mode == "weights_free" else None
    )
    set_global(server)
    return server


def cases(args):
    return [
        json.loads(p.read_text())
        for p in sorted(
            (args.output / "cases").glob("*/streaming-case.json"),
            key=lambda p: (json.loads(p.read_text())["height"], p.parent.name),
        )
    ]


def references(paths, ids):
    from continuous_source_media import reference_payload

    return [
        dict(type="image", **reference_payload(p, i))
        for p, i in zip(paths, ids, strict=True)
    ]


def encode(args):
    from sglang.multimodal_gen.runtime.conditioning.native import build_encoder_pipeline
    from sglang.multimodal_gen.runtime.conditioning.worker import LocalConditioner
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.codecs import (
        ZingH3Codecs,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.encoder_worker import (
        load_encoder,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.media_geometry import (
        resolve_source_geometry,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_encoder import (
        create_source_buffer,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.source_session import (
        source_preview,
    )

    server = server_args(args)
    with tempfile.TemporaryDirectory() as rendezvous:
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=1,
            sp_size=1,
            distributed_init_method="file://" + rendezvous + "/group",
        )
        qwen = build_encoder_pipeline(str(args.model), server)
        source = load_encoder(args.model)
        server.pipeline_config.vae_config.arch_config = source.va
        codecs = ZingH3Codecs(
            {**qwen.modules, **source.modules}, server.pipeline_config, source.runtime
        )
        profile = json.loads((args.model / "zing_realtime_profile.json").read_text())
        source_profile = dict(profile["source_encoder"])
        source_profile.pop("remote", None)
        for case in cases(args):
            root = args.output / "encoded" / case["label"]
            root.mkdir(parents=True, exist_ok=False)
            condition = LocalConditioner(
                codecs.encode_condition, "frozen-review", cache_bytes=256 << 20
            )
            sw, sh = (
                case.get("source_width", case["width"]),
                case.get("source_height", case["height"]),
            )
            geometry = resolve_source_geometry(
                (sh, sw),
                (case["height"], case["width"]),
                server.pipeline_config,
                dict(
                    max_short_edge=codecs.max_short_edge,
                    max_area_config=codecs.max_area_config,
                    max_source_target_aspect_ratio_factor=codecs.max_source_target_aspect_ratio_factor,
                ),
            )
            encoder = create_source_buffer(codecs, source_profile, capacity_frames=34)
            decoder = subprocess.Popen(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    case["source_video"],
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-vsync",
                    "0",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
            )
            rows, cursor, event_id = [], 0, 1
            updates = iter(case.get("prompt_updates", []))
            pending = next(updates, None)
            inputs = None

            def record(message):
                name = f"message-{len(rows):05d}.pt"
                save(root / name, message)
                rows.append(name)

            def publish():
                record(
                    dict(
                        type="conditioning",
                        tensors=inputs.positive,
                        inputs={
                            f.name: getattr(inputs, f.name)
                            for f in fields(inputs)
                            if f.name
                            not in ("positive", "source_visual", "condition_media")
                        },
                        event_id=event_id,
                        prompt_version=event_id,
                        **({"source_boundary": cursor} if cursor else {}),
                    )
                )

            def drain():
                while encoder.ready:
                    item = encoder.pop()
                    clip, offset = divmod(item.end - 1, 5)
                    record(
                        dict(
                            type="source.moments",
                            tensor=item.values,
                            latent_start=item.start,
                            latent_end=item.end,
                            pixel_end=min(case["frames"], clip * 17 + 1 + offset * 4),
                            source_dependency_end=item.source_dependency_end,
                        )
                    )

            try:
                with (
                    torch.inference_mode(),
                    set_forward_context(current_timestep=0, attn_metadata=None),
                ):
                    while cursor < case["frames"]:
                        count = min(17, case["frames"] - cursor)
                        raw = decoder.stdout.read(count * sw * sh * 3)
                        if len(raw) != count * sw * sh * 3:
                            raise ValueError(
                                "Source video ended before frozen frame count"
                            )
                        if inputs is None:
                            refs = references(case["references"], case["reference_ids"])
                            request = dict(
                                messages=[
                                    dict(
                                        role="user",
                                        content=[
                                            *refs,
                                            dict(type="text", text=case["prompt"]),
                                        ],
                                    ),
                                    dict(
                                        role="target",
                                        content=[
                                            dict(
                                                type="video",
                                                output=dict(
                                                    frames=17,
                                                    width=case["width"],
                                                    height=case["height"],
                                                ),
                                            )
                                        ],
                                        conditions=[
                                            dict(
                                                content_id=r["id"],
                                                role="reference",
                                                latent_modalities=["visual"],
                                            )
                                            for r in refs
                                        ],
                                    ),
                                ]
                            )
                            inputs = codecs.prepare(
                                request,
                                lambda _: nullcontext(),
                                retain_condition_media=True,
                                source_preview=source_preview(
                                    dict(width=sw, height=sh, fps=24), raw
                                ),
                                conditioning=condition,
                            )
                            publish()
                        if (
                            pending is not None
                            and pending["source_frame_start"] == cursor
                        ):
                            event_id += 1
                            refs = pending["references"]
                            inputs = codecs.prepare_reference_update(
                                inputs,
                                references(
                                    [r["path"] for r in refs], [r["id"] for r in refs]
                                ),
                                pending["prompt"],
                                lambda _: nullcontext(),
                                conditioning=condition,
                            )
                            publish()
                            pending = next(updates, None)
                        pixels = (
                            torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                            .reshape(count, sh, sw, 3)
                            .permute(3, 0, 1, 2)
                        )
                        encoder.append(geometry.resize(pixels), frame_start=cursor)
                        cursor += count
                        drain()
                    encoder.finish()
                    drain()
                    if pending is not None or decoder.stdout.read(1):
                        raise ValueError(
                            "Source boundaries or frame count differ from frozen case"
                        )
                    if decoder.wait() != 0:
                        raise RuntimeError("Source FFmpeg failed")
                    record(
                        dict(type="source.end", latent_frames=encoder.emitted_latents)
                    )
            finally:
                encoder.dispose()
                if decoder.poll() is None:
                    decoder.kill()
                decoder.wait()
            write(
                root / "COMPLETE.json",
                dict(
                    messages=rows,
                    case_id=case["case_id"],
                    geometry=geometry.receipt(),
                    scope="untimed native Encoder preparation; exact 17-frame source barriers",
                ),
            )
            print("ENCODED " + case["label"], flush=True)


def generate(args):
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.adapters.zing_h3_source_adapter import (
        _source_sampling_schedule,
        _with_source_pipeline_policy,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
        TransformerLoader,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.dit_worker import (
        create_link_session,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )

    server = server_args(args)
    with tempfile.TemporaryDirectory() as rendezvous:
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=1,
            sp_size=1,
            distributed_init_method="file://" + rendezvous + "/group",
        )
        model, _ = TransformerLoader().load(
            str(args.model / "transformer"),
            server,
            "transformer",
            "diffusers",
            allow_native_fallback=False,
        )
        model.eval()
        runtime = json.loads((args.model / "zing_config.json").read_text())
        profile = _with_source_pipeline_policy(
            json.loads((args.model / "zing_realtime_profile.json").read_text()),
            joint_forward=server.zing_source_joint_forward,
        )
        server.pipeline_config.vae_config.update_model_arch(
            json.loads((args.model / "video_vae/config.json").read_text())
        )
        stage = ZingH3GenerationStage(
            {"transformer": model},
            server.pipeline_config,
            runtime,
            external_inputs=True,
        )
        stage.deployment_sampling = _source_sampling_schedule(server, profile)
        params = SimpleNamespace(
            num_inference_steps=2,
            guidance_scale=profile["guidance_scale"],
            zing_joint_forward=profile.get("joint_forward", "cross_chunk"),
            zing_window_chunks=6,
            zing_sink_chunks=3,
            zing_window_eviction_timing="after_step0",
            zing_rope_position_mode="block_relative",
            zing_rope_max_frame_gap=12,
        )
        for case in cases(args):
            source = args.output / "encoded" / case["label"]
            metadata = json.loads((source / "COMPLETE.json").read_text())
            dest = args.output / "latents" / case["label"]
            if dest.exists() and any(dest.iterdir()):
                raise ValueError("Partial phase output must be archived before retry")
            dest.mkdir(parents=True, exist_ok=args.resume)
            count, frames = 0, 0
            hashes = []

            async def output(result):
                nonlocal count, frames
                if not torch.isfinite(result.visual).all():
                    raise ValueError("Generated DiT latent contains nonfinite values")
                path = dest / f"chunk-{count:05d}.pt"
                save(
                    path,
                    dict(
                        visual=result.visual,
                        chunk=result.chunk,
                        provenance=result.provenance,
                        event_id=result.prompt_event_id,
                        prompt_version=result.prompt_version,
                    ),
                )
                hashes.append(
                    dict(
                        file=path.name,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        bytes=path.stat().st_size,
                    )
                )
                frames = result.provenance.pixels.end
                count += 1
                if count % 25 == 0:
                    print(
                        f"LATENTS {case['label']} chunks={count} frames={frames}",
                        flush=True,
                    )

            sampler = stage._build_realtime_sampler(params)
            sampler.prompt_joint = True
            init = dict(
                session_id=case["label"],
                generation_id="frozen-123",
                seed=123,
                width=case["width"],
                height=case["height"],
                fps=24,
            )
            worker = create_link_session(
                sampler, stage.codecs, init, output, defer_final=True
            )
            worker.source_records = [(1, 0, case["frames"])]

            async def run():
                task = asyncio.create_task(worker.run())
                try:
                    for name in metadata["messages"]:
                        row = map_tensors(load(source / name), lambda t: t.cuda())
                        await worker.put(
                            dict(
                                row,
                                session_id=init["session_id"],
                                generation_id=init["generation_id"],
                            )
                        )
                        while worker.queue or worker.in_flight:
                            if task.done():
                                await task
                            await asyncio.sleep(0.005)
                    await task
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    worker.close()

            with (
                torch.inference_mode(),
                set_forward_context(current_timestep=0, attn_metadata=None),
            ):
                asyncio.run(run())
            if frames != case["frames"]:
                raise ValueError(
                    f"Generated frame coverage differs: {frames} != {case['frames']}"
                )
            write(
                dest / "COMPLETE.json",
                dict(
                    chunks=count,
                    frames=frames,
                    files=hashes,
                    source_sha=args.source_sha,
                    sampling=dict(
                        seed=123, schedule=stage.deployment_sampling, window=6, sink=3
                    ),
                    scope="frozen generated DiT latents; offline dynamic KV, no serving performance claim",
                ),
            )
            print("GENERATED " + case["label"], flush=True)


def decode(args):
    # Compare the explicit opt-out with the actual serving default, not an
    # explicit opt-in that could hide a broken default after a policy change.
    if args.cell == "decoder-sm90":
        os.environ.pop("SGLANG_LYNN_SM90_FAST_DECODE", None)
    else:
        os.environ["SGLANG_LYNN_SM90_FAST_DECODE"] = "0"
    from sglang.multimodal_gen.configs.pipeline_configs.zing_h3_decoder import (
        normalize_decoder_spec,
    )
    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.loader import (
        load_lynn_decoder,
    )

    spec = normalize_decoder_spec(
        dict(
            type="lynn_streaming",
            checkpoint_path=str(args.lynn / "decoder.pt"),
            model_dir=str(args.lynn),
            precision="bf16",
            weights="ema",
        ),
        video_decoder_backend=None,
        vae_decode_precision="fp32",
    )
    model = load_lynn_decoder(spec, device="cuda").eval()
    if model.architecture["decoder_num_layers"] != 26:
        raise ValueError("Product review requires the production 26-layer Lynn")
    for case in cases(args):
        source = args.output / "latents" / case["label"]
        receipt = json.loads((source / "COMPLETE.json").read_text())
        dest = args.output / args.cell / case["label"]
        dest.mkdir(parents=True, exist_ok=False)
        counts_before = dict(model.execution.counts)
        runs = []
        for index in range(4):
            capture = (
                MediaCapture(
                    dest, case["width"], case["height"], case["frames"], mode="full"
                )
                if index == 1
                else None
            )
            stream = model.new_stream(frame_limit=case["frames"])
            torch.cuda.synchronize()
            resident = dict(
                allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),
            )
            torch.cuda.reset_peak_memory_stats()
            intervals, emitted = [], 0
            rgb_hash = hashlib.sha256()
            try:
                with torch.inference_mode():
                    for row in receipt["files"]:
                        path = source / row["file"]
                        if (
                            hashlib.sha256(path.read_bytes()).hexdigest()
                            != row["sha256"]
                        ):
                            raise ValueError("Frozen latent hash changed")
                        value = load(path)["visual"].cuda()
                        if not torch.isfinite(value).all():
                            raise ValueError(
                                "Frozen DiT latent contains nonfinite values"
                            )
                        a, b = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        a.record()
                        rgb = stream.push(value)
                        b.record()
                        b.synchronize()
                        intervals.append(
                            dict(
                                host_ms=(time.perf_counter() - start) * 1000,
                                cuda_ms=a.elapsed_time(b),
                                frames=rgb.shape[2],
                            )
                        )
                        if not torch.isfinite(rgb).all():
                            raise ValueError("Decoder output contains nonfinite values")
                        emitted += rgb.shape[2]
                        if rgb.shape[2]:
                            payload = (
                                rgb[0]
                                .permute(1, 2, 3, 0)
                                .mul(255)
                                .round()
                                .clamp_(0, 255)
                                .to(torch.uint8)
                                .cpu()
                                .contiguous()
                                .numpy()
                                .tobytes()
                            )
                            rgb_hash.update(payload)
                            if capture is not None:
                                capture.write(payload)
                    stream.finish(case["frames"])
                    if emitted != case["frames"]:
                        raise ValueError("Decoder changed frame coverage")
                    runs.append(
                        dict(
                            phase="warmup" if index == 0 else "formal",
                            pushes=intervals,
                            rgb_sha256=rgb_hash.hexdigest(),
                            resident=resident,
                            peak=dict(
                                allocated=torch.cuda.max_memory_allocated(),
                                reserved=torch.cuda.max_memory_reserved(),
                            ),
                        )
                    )
            finally:
                stream.close()
            if capture is not None:
                write(dest / "media.json", capture.finish())
        engine = model.execution.sm90_engine
        counts = {k: v - counts_before[k] for k, v in model.execution.counts.items()}
        if args.cell == "decoder-sm90" and (
            counts["sm90_fast_decode"] == 0 or engine is None or engine.fast_start != 4
        ):
            raise ValueError(
                "Product review did not exercise the 26-layer SM90 deep path"
            )
        if args.cell == "decoder-block" and counts["sm90_fast_decode"]:
            raise ValueError("Baseline unexpectedly used the SM90 path")
        active = [
            r["cuda_ms"] for run in runs[1:] for r in run["pushes"] if r["frames"]
        ]
        write(
            dest / "summary.json",
            dict(
                source_sha=args.source_sha,
                case_id=case["case_id"],
                switch_environment=os.environ.get("SGLANG_LYNN_SM90_FAST_DECODE"),
                execution_options=asdict(model.execution.options),
                execution_counts=counts,
                fast_start=None if engine is None else engine.fast_start,
                decoder_architecture=model.architecture,
                runs=runs,
                sample_count=len(active),
                median_ms=statistics.median(active),
                p95_ms=sorted(active)[int(0.95 * (len(active) - 1))],
                frozen_latents=receipt,
                scope="isolated decoder push timing; disk IO, H2D and video encoding excluded; not E2E",
            ),
        )
        print("DECODED " + args.cell + " " + case["label"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "lynn", "media-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--adaln-table", type=Path)
    parser.add_argument(
        "--adaln-mode", choices=("weights_free", "resident"), default="weights_free"
    )
    parser.add_argument("--source-sha", required=True)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="CPU only: validate assets and freeze all eight case descriptions",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed owned phases; never overwrite partial phase outputs",
    )
    parser.add_argument(
        "--phase", choices=("all", "encode", "generate", "decode"), default="all"
    )
    parser.add_argument("--cell", choices=("decoder-block", "decoder-sm90"))
    args = parser.parse_args()
    if args.prepare_only and args.phase != "all":
        parser.error("--prepare-only requires --phase all")
    if args.phase != "all" and len(cases(args)) != 8:
        parser.error("Individual phases require all eight prepared cases")
    if args.adaln_mode == "weights_free" and args.adaln_table is None:
        parser.error("weights_free requires --adaln-table")
    if args.phase == "all":
        expected = os.environ.get("PR693_EXPECTED_GPU")
        if expected and not args.prepare_only:
            visible = (
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                    text=True,
                )
                .strip()
                .splitlines()
            )
            if visible != [expected]:
                raise RuntimeError("Review GPU identity differs from reservation")
            if subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True,
            ).strip():
                raise RuntimeError("Review GPU became occupied before execution")
        if not args.prepare_only:
            memory = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                .strip()
                .splitlines()
            )
            frozen = list((args.output / "latents").glob("*/COMPLETE.json"))
            if len(memory) != 1:
                raise RuntimeError("Frozen review requires exactly one visible GPU")
            if int(memory[0]) <= 90000 and len(frozen) != 8:
                raise RuntimeError(
                    "Full product DiT generation exceeded the measured 80-GiB H100 capacity. Prepare/encode separately, generate on a large-memory worker, then resume Decoder A/B with all eight frozen latent receipts."
                )
        args.output.mkdir(parents=True, exist_ok=args.resume)
        manifest = Path(__file__).parent / "case_sets/product-case-20261103.json"
        for entry in json.loads(manifest.read_text())["cases"]:
            label = entry["case_id"].removeprefix("zing-product-case-20261103-")
            if (
                args.resume
                and (args.output / "cases" / label / "streaming-case.json").is_file()
            ):
                continue
            prepare_case(
                label,
                args.media_root,
                args.output / "cases" / label,
                manifest=Path(__file__).resolve().parents[2] / entry["manifest"],
                case_id=entry["case_id"],
            )
        if not (args.output / "experiment.json").exists():
            write(
                args.output / "experiment.json",
                dict(
                    source_sha=args.source_sha,
                    method="one generated latent stream per case, identical frozen tensors for decoder A/B",
                    e2e_verified=False,
                ),
            )
        if args.prepare_only:
            print("PREPARED_ALL_EIGHT_CASES_CPU_ONLY", flush=True)
            return
        for phase, cell in (
            ("encode", None),
            ("generate", None),
            ("decode", "decoder-block"),
            ("decode", "decoder-sm90"),
        ):
            phase_root = (
                "encoded"
                if phase == "encode"
                else "latents"
                if phase == "generate"
                else cell
            )
            marker = "summary.json" if cell else "COMPLETE.json"
            prepared = cases(args)
            if len(prepared) != 8:
                raise ValueError("Resume requires the complete eight-case inventory")
            if args.resume and all(
                (args.output / phase_root / c["label"] / marker).is_file()
                for c in prepared
            ):
                print("REUSED " + phase_root, flush=True)
                continue
            command = [sys.executable, "-u", __file__, *sys.argv[1:], "--phase", phase]
            env = dict(os.environ)
            if cell:
                command += ["--cell", cell]
                if cell == "decoder-sm90":
                    env.pop("SGLANG_LYNN_SM90_FAST_DECODE", None)
                else:
                    env["SGLANG_LYNN_SM90_FAST_DECODE"] = "0"
            subprocess.run(command, env=env, check=True)
            receipt_path = args.output / "experiment.json"
            receipt = json.loads(receipt_path.read_text())
            receipt.setdefault("phase_sources", {})[phase_root] = args.source_sha
            write(receipt_path, receipt)
        write(
            args.output / "COMPLETE.json",
            dict(complete=True, source_sha=args.source_sha),
        )
    else:
        # Models and JIT objects can create large reference cycles at interpreter
        # shutdown. Disable automatic GC within this short-lived worker, explicitly
        # close distributed groups, flush receipts, then let process exit release
        # all GPU allocations without Python's unbounded teardown walk.
        gc.disable()
        try:
            {"encode": encode, "generate": generate, "decode": decode}[args.phase](args)
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                cleanup_dist_env_and_memory,
            )

            cleanup_dist_env_and_memory()
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
