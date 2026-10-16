# SPDX-License-Identifier: Apache-2.0
"""torchrun: isolated DiT replay of trusted, locally captured source chunks.

Capture files are pickle-bearing experiment artifacts; never accept untrusted
uploads. All input transfers happen before timing. Each repetition has fresh KV
and RNG state. No codec or media IO executes within the measured region.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.component_capture import (
    map_tensors,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
    SourceReadySession,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    ZingH3Packer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.sampling import (
    ZingH3Sampler,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-runs", type=int, default=5)
    args = parser.parse_args()
    world, rank, local = (
        int(os.environ[k]) for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK")
    )
    torch.cuda.set_device(local)
    from sglang.multimodal_gen.runtime.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
        TransformerLoader,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )

    init_distributed_environment(world_size=world, rank=rank, local_rank=local)
    initialize_model_parallel(sequence_parallel_degree=world, ulysses_degree=world)
    server = ServerArgs.from_kwargs(
        model_path=str(args.model),
        num_gpus=world,
        sp_degree=world,
        ulysses_degree=world,
        ring_degree=1,
        video_decoder_backend="native",
        encoder_parallel="replicate",
        attention_backend="torch_sdpa",
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        use_fsdp_inference=False,
        performance_mode="speed",
        warmup_mode="off",
    )
    set_global_server_args(server)
    model, _ = TransformerLoader().load(
        str(args.model / "transformer"),
        server,
        "transformer",
        "diffusers",
        allow_native_fallback=False,
    )
    model.eval()
    out = args.output / f"rank{rank}"
    out.mkdir(parents=True, exist_ok=False)
    count = json.loads((args.capture / "COMPLETE.json").read_text())["chunks"]
    session = torch.load(
        args.capture / "session.pt", map_location="cpu", weights_only=False
    )
    frozen = [
        torch.load(
            args.capture / f"chunk-{i:05d}.pt", map_location="cpu", weights_only=False
        )
        for i in range(count)
    ]
    session = map_tensors(session, lambda t: t.to("cuda"))
    frozen = map_tensors(frozen, lambda t: t.to("cuda"))
    runtime, params = session["runtime"], session["params"]
    packer = ZingH3Packer(
        session["arch"],
        num_frame_per_block=runtime["num_frame_per_block"],
        split_first_latent=runtime["split_first_latent"],
        **{
            key: runtime["rope_config"][key]
            for key in ("spatial_extent", "frame_scale", "visual_token_frame_spans")
        },
    )

    def new_sampler():
        return ZingH3Sampler(
            model,
            packer,
            solver=params.zing_solver,
            sigma_points=params.num_inference_steps,
            denoising_step_list=params.zing_denoising_step_list,
            video_shift=runtime["timestep_shift"],
            audio_shift=runtime["audio_timestep_shift"],
            num_train_timesteps=runtime["num_train_timestep"],
            guidance_scale=params.guidance_scale,
            window_chunks=params.zing_window_chunks,
            sink_chunks=params.zing_sink_chunks,
            rope_position_mode=params.zing_rope_position_mode,
            rope_max_frame_gap=params.zing_rope_max_frame_gap,
            window_eviction_timing=params.zing_window_eviction_timing,
        )

    @torch.no_grad()
    def run(capture=False):
        sampler = new_sampler()
        forward_events = []

        def pre(module, inputs):
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            forward_events.append((a, b))

        def post(module, inputs, output):
            forward_events[-1][1].record()

        before = model.register_forward_pre_hook(pre)
        after = model.register_forward_hook(post)
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        state = SourceReadySession(
            sampler,
            session["inputs"],
            seed=session["seed"],
            max_pending_chunks=1,
            max_pending_bytes=256 << 20,
        )
        values, events = [], []
        try:
            with set_forward_context(current_timestep=0, attn_metadata=None):
                for entry in frozen:
                    chunk = entry["chunk"]
                    a, b = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    a.record()
                    state.append(
                        chunk.source,
                        chunk.noise,
                        provenance=chunk.provenance,
                        final=chunk.final,
                    )
                    result = state.step()
                    b.record()
                    events.append((a, b))
                    if capture:
                        values.append(result.visual)
                if not state.ended:
                    state.finish()
                assert state.complete
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            stats = dict(
                seconds=seconds,
                block_ms=[a.elapsed_time(b) for a, b in events],
                transformer_ms=sum(a.elapsed_time(b) for a, b in forward_events),
                forward_count=len(forward_events),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            return stats, values
        finally:
            before.remove()
            after.remove()
            state.dispose()

    warmup, actual = run(capture=True)
    # Do not infer an SP tolerance. Keep raw outputs and descriptive errors;
    # strict replay status is separate from timing completeness.
    comparison = []
    for got, row in zip(actual, frozen):
        expected = row["expected_visual"]
        delta = got.float() - expected.float()
        if not torch.isfinite(got).all() or got.shape != expected.shape:
            raise RuntimeError("DiT replay produced invalid output")
        comparison.append(
            dict(
                exact=bool(torch.equal(got, expected)), max_abs=delta.abs().max().item()
            )
        )
    torch.save([v.cpu() for v in actual], out / "replay-latents.pt")
    (out / "replay-comparison.json").write_text(json.dumps(comparison))
    del actual
    runs = []
    for _ in range(args.measured_runs):
        stats, _ = run()
        runs.append(stats)
        (out / "measured.json").write_text(json.dumps(runs, indent=2))
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        profile, _ = run()
    prof.export_chrome_trace(str(out / "dit.trace.json"))
    seconds = [x["seconds"] for x in runs]
    (out / "summary.json").write_text(
        json.dumps(
            dict(
                sp=world,
                rank=rank,
                median=statistics.median(seconds),
                min=min(seconds),
                max=max(seconds),
                warmup=warmup,
                profile_seconds_separate=profile["seconds"],
                strict_replay_passed=all(x["exact"] for x in comparison),
                stable=(max(seconds) - min(seconds)) / statistics.median(seconds)
                <= 0.1,
            ),
            indent=2,
        )
    )
    (out / "COMPLETE").write_text(
        "Timing and raw replay outputs collected; inspect numerical status\n"
    )


if __name__ == "__main__":
    main()
