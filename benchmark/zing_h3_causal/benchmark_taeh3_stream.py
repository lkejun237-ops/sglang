# SPDX-License-Identifier: Apache-2.0
"""Measure finite H3 streaming through the real scheduler and RGB transport.

This measures worker generation, GPU-to-CPU materialization and local IPC. It
excludes WebSocket/network delivery and browser playback. No output pacing is
applied, so deadline misses cannot be hidden by a fixed-rate exporter.
"""

import argparse
import copy
import json
import math
import os
import statistics
import time
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sp-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=1440)
    parser.add_argument(
        "--transport", choices=("scheduler", "websocket"), default="scheduler"
    )
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument(
        "--conditioning-mode", choices=("local", "remote"), default="local"
    )
    parser.add_argument("--conditioning-service-url")
    parser.add_argument("--conditioning-profile")
    parser.add_argument("--conditioning-config")
    args = parser.parse_args()
    conditioning = {}
    if args.conditioning_mode != "local":
        conditioning = dict(
            zing_conditioning_mode=args.conditioning_mode,
            conditioning_service_url=args.conditioning_service_url,
            conditioning_expected_profile=args.conditioning_profile,
            conditioning_config=args.conditioning_config,
        )
    os.environ.setdefault("SGLANG_ZING_EXPERIMENTAL_ATTENTION", "fa4")
    os.environ.setdefault("SGLANG_ZING_EXPERIMENTAL_STABLE_SHAPES", "1")
    os.environ.setdefault("SGLANG_ZING_EXPERIMENTAL_COMPILED_ADALN", "1")
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
        ReleaseRealtimeSessionReq,
    )
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
    from sglang.multimodal_gen.runtime.realtime.zing_chunk_plan import ChunkPlan
    from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    request = json.loads(Path(args.request).read_text())
    target = request["messages"][1]["content"][0]
    if "source" in request["messages"][1]:
        raise ValueError(
            "Do not lengthen a source video without a matching source asset"
        )
    target["type"] = "video"
    prompt = request["messages"][0]["content"][0]["text"]
    runtime = json.loads((Path(args.model) / "zing_config.json").read_text())
    plan = ChunkPlan(runtime["num_frame_per_block"], runtime["split_first_latent"])
    started = time.perf_counter()
    generator = DiffGenerator.from_pretrained(
        model_path=args.model,
        num_gpus=args.sp_size,
        sp_degree=args.sp_size,
        ulysses_degree=args.sp_size,
        ring_degree=1,
        encoder_parallel="replicate",
        local_mode=True,
        attention_backend="torch_sdpa",
        use_fsdp_inference=False,
        dit_layerwise_offload=False,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        performance_mode="speed",
        pin_cpu_memory=False,
        vae_cpu_offload=False,
        warmup_mode="off",
        pipeline_config={
            "video_decoder_backend": "taeh3",
            "taeh3_checkpoint_path": args.weights,
            "taeh3_decode_precision": args.dtype,
        },
        **conditioning,
    )
    report = {
        "load_seconds": time.perf_counter() - started,
        "sp_size": args.sp_size,
        "fps": 24,
        "frames": args.frames,
        "dtype": args.dtype,
        "conditioning_mode": args.conditioning_mode,
        "scope": __doc__,
        "runs": [],
    }
    try:
        if args.transport == "websocket":
            from benchmark_taeh3_websocket import run_websocket_benchmark

            run_websocket_benchmark(generator, request, output, args.frames)
            return
        # First session is a cold 10s request, then a fresh warm long session.
        for label, frames in (("cold_10s", 240), ("warm_sustained", args.frames)):
            req_json = copy.deepcopy(request)
            req_json["messages"][1]["content"][0]["output"]["frames"] = frames
            full, tail = divmod(frames - 1, 17)
            latents = full * 5 + 1 + math.ceil(tail / 4)
            chunks = plan.count(latents)
            session_id = str(uuid.uuid4())
            rows, emitted, first_ready, play_until, stalls = [], 0, None, None, 0.0
            began = time.perf_counter()
            try:
                for index in range(chunks):
                    params = SamplingParams.from_user_sampling_params_args(
                        args.model,
                        server_args=generator.server_args,
                        prompt=prompt,
                        num_frames=frames,
                        height=target["output"]["height"],
                        width=target["output"]["width"],
                        fps=24,
                        num_inference_steps=4,
                        guidance_scale=1.0,
                        seed=42,
                        zing_request=req_json,
                        zing_solver="few_step",
                        zing_denoising_step_list=[1000, 750, 500, 250],
                        zing_realtime_decode_mode="stable",
                        save_output=False,
                    )
                    batch = prepare_request(
                        server_args=generator.server_args, sampling_params=params
                    )
                    batch.realtime_session_id = session_id
                    batch.realtime_generation_id = session_id
                    batch.block_idx = index
                    batch.return_raw_frames = True
                    batch.realtime_output_pacing = False
                    tick = time.perf_counter()
                    result = generator._send_to_scheduler_and_wait_for_response([batch])
                    elapsed = time.perf_counter() - began
                    if result.error:
                        raise RuntimeError(result.error)
                    count = sum(
                        len(group) for group in (result.raw_frame_batches or [])
                    )
                    stall = 0.0
                    if count:
                        if first_ready is None:
                            first_ready = play_until = elapsed
                        stall = max(0.0, elapsed - play_until)
                        stalls += stall
                        play_until = max(play_until, elapsed) + count / 24
                        emitted += count
                    rows.append(
                        {
                            "block": index,
                            "arrival_seconds": elapsed,
                            "call_seconds": time.perf_counter() - tick,
                            "frames": count,
                            "cumulative_frames": emitted,
                            "pts_start_seconds": (
                                (emitted - count) / 24 if count else None
                            ),
                            "pts_end_seconds": emitted / 24 if count else None,
                            "stall_seconds": stall,
                            "unadjusted_playback_buffer_seconds": (
                                None
                                if first_ready is None
                                else emitted / 24 - (elapsed - first_ready)
                            ),
                            "metadata": result.raw_frame_metadata,
                            "usage": result.usage,
                        }
                    )
                    del result
                assert emitted == frames, (emitted, frames)
                duration = time.perf_counter() - began
                latency = sorted(row["call_seconds"] for row in rows)
                summary = {
                    "name": label,
                    "frames": frames,
                    "total_seconds": duration,
                    "first_frame_seconds": first_ready,
                    "fps_including_startup": frames / duration,
                    "rtf": duration / (frames / 24),
                    "gpu_seconds": duration * args.sp_size,
                    "chunk_p50_seconds": statistics.median(latency),
                    "chunk_p95_seconds": latency[
                        min(len(latency) - 1, math.ceil(0.95 * len(latency)) - 1)
                    ],
                    "playback_stall_seconds": stalls,
                    "request": req_json,
                    "steady_deadline_pass": stalls <= 0.001,
                    "chunks": len(rows),
                }
                report["runs"].append(summary)
                (output / "stream-report.json").write_text(json.dumps(report, indent=2))
                print(json.dumps(summary), flush=True)
            finally:
                (output / f"{label}-chunks.json").write_text(
                    json.dumps(rows, indent=2, default=str)
                )
                sync_scheduler_client.forward(
                    ReleaseRealtimeSessionReq(session_id=session_id)
                )
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
