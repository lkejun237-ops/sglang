# SPDX-License-Identifier: Apache-2.0
"""torchrun: isolated DiT replay of trusted, locally captured source chunks.

Capture files are pickle-bearing experiment artifacts; never accept untrusted
uploads. All input transfers happen before timing. Each repetition has fresh KV
and RNG state. No codec or media IO executes within the measured region.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from gc_pause_observer import GCPauseObserver
from resident_soak import (
    all_ranks_reached_duration,
    replay_collection_decision,
    run_resident_round,
)

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.component_capture import (
    map_tensors,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.incremental import (
    SourceReadySession,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.packing import (
    ZingH3Packer,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument(
        "--minimum-round-seconds",
        type=float,
        default=0,
        help="Additional resident soak: repeat complete sessions until each round reaches this wallclock duration; zero retains one complete session per round",
    )
    parser.add_argument(
        "--collect-soak-on-numerical-mismatch",
        action="store_true",
        help="Collect diagnostic resident soak despite bitexact failure; retain numerical FAIL and never mark full acceptance",
    )
    parser.add_argument(
        "--queued-encoder",
        action="store_true",
        help="Validate the DiT input queue using captured Encoder moments and Qwen features",
    )
    parser.add_argument(
        "--execution-context",
        choices=["inference", "no_grad"],
        default="inference",
        help="Production executes under inference_mode; no_grad is an explicit diagnostic control",
    )
    parser.add_argument(
        "--diagnostic-blocks",
        type=int,
        default=None,
        help="Supplemental prefix-only numerical probe; exits after comparison, never writes COMPLETE or formal timing",
    )
    parser.add_argument("--pipeline-config", type=Path)
    parser.add_argument("--require-captured-kv-pool", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Additional diagnostic round, excluded from formal metrics",
    )
    args = parser.parse_args()
    if args.measured_runs < 1:
        parser.error("measured-runs must be positive")
    if not math.isfinite(args.minimum_round_seconds) or args.minimum_round_seconds < 0:
        parser.error("minimum-round-seconds must be finite and nonnegative")
    if args.diagnostic_blocks is not None and (
        args.diagnostic_blocks < 1
        or args.minimum_round_seconds
        or args.queued_encoder
        or args.profile
    ):
        parser.error(
            "Diagnostic prefix requires positive blocks and cannot run soak, queued replay or profile"
        )
    if args.collect_soak_on_numerical_mismatch and (
        not args.minimum_round_seconds
        or args.diagnostic_blocks is not None
        or args.queued_encoder
        or args.profile
    ):
        parser.error("Mismatch collection requires a complete resident soak only")
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
    if args.pipeline_config:
        for key, value in json.loads(args.pipeline_config.read_text()).items():
            if not hasattr(server.pipeline_config, key):
                raise ValueError(f"Unknown pipeline setting: {key}")
            setattr(server.pipeline_config, key, value)
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
    (out / "EXECUTION_CONTEXT.json").write_text(
        json.dumps(
            dict(
                requested=args.execution_context,
                model_load_inference_mode=torch.is_inference_mode_enabled(),
                scope="Sampling session enters the selected context; model loading stays outside",
            ),
            indent=2,
        )
    )
    capture_metadata = json.loads((args.capture / "COMPLETE.json").read_text())
    count = capture_metadata["chunks"]
    session = torch.load(
        args.capture / "session.pt", map_location="cpu", weights_only=False
    )
    replay_count = (
        count if args.diagnostic_blocks is None else min(count, args.diagnostic_blocks)
    )
    frozen = [
        torch.load(
            args.capture / f"chunk-{i:05d}.pt", map_location="cpu", weights_only=False
        )
        for i in range(replay_count)
    ]
    session = map_tensors(session, lambda t: t.to("cuda"))
    # Expected outputs stay on CPU; only ready inputs belong in the timed GPU working set.
    frozen = [
        {**row, "chunk": map_tensors(row["chunk"], lambda t: t.to("cuda"))}
        for row in frozen
    ]
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

    policy = session.get("replay_policy")
    if not policy:
        raise ValueError(
            "Capture lacks resolved production schedule; recapture with the current server"
        )
    if policy.get("work_quantum") != 1:
        raise ValueError("This replay requires captured production work_quantum=1")
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.realtime_sampling import (
        build_source_realtime_sampler,
    )

    prompts = {}
    for entry in frozen:
        name = entry.get("prompt_file")
        if name and name not in prompts:
            prompts[name] = map_tensors(
                torch.load(args.capture / name, map_location="cpu", weights_only=False),
                lambda t: t.to("cuda"),
            )

    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.zing_h3.stage import (
        ZingH3GenerationStage,
    )
    from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
        ZingKVGeometry,
        ZingKVPool,
        ZingKVPoolConfig,
        ZingKVPoolPlan,
    )

    pool = None
    captured_pool = session.get("kv_pool")
    if captured_pool is None and args.require_captured_kv_pool:
        raise ValueError(
            "Capture lacks the configured production KV pool; recapture before replay"
        )
    if captured_pool is not None:
        if captured_pool.get("schema_version") != 2:
            raise ValueError(
                "KV capture lacks the current memory policy; recapture before replay"
            )
        if captured_pool["world_size"] != world:
            raise ValueError(
                "Replay SP degree differs from the captured production pool"
            )
        from sglang.multimodal_gen.runtime.utils.precision_types import (
            PRECISION_TO_TYPE,
        )

        geometry = ZingKVGeometry(
            **captured_pool["geometry"],
            chunk_plan=packer.chunk_plan,
            video_t_span=packer._video_t_span,
        )
        plan = ZingKVPoolPlan.build(
            ZingKVPoolConfig(**captured_pool["config"]),
            geometry,
            token_capacity=captured_pool["slot_rows"],
            **captured_pool["runtime_policy"],
        )
        if plan.slot_rows != captured_pool["slot_rows"]:
            raise ValueError("Reconstructed KV slot capacity differs from capture")
        pool = ZingKVPool.allocate(
            plan,
            num_layers=session["arch"].num_layers,
            local_heads=session["arch"].num_attention_heads // world,
            head_dim=session["arch"].attention_head_dim,
            dtype=PRECISION_TO_TYPE[captured_pool["dtype"]],
            device=torch.device("cuda", local),
        )
        (out / "KV_POOL.json").write_text(
            json.dumps(
                dict(
                    slot_rows=plan.slot_rows,
                    allocated_bytes=pool.allocated_bytes(),
                    sp=world,
                )
            )
        )

    def new_sampler():
        sampler = build_source_realtime_sampler(
            model, packer, runtime, params, policy["schedule"]
        )
        # Source-session admission also binds this resolved policy on the
        # sampler. Keep replay metadata and later prompt handling identical.
        sampler.prompt_joint = policy["prompt_joint"]
        return sampler

    import atexit
    import gc

    from sglang.multimodal_gen.runtime.utils.zing_static_gc import prepare_static_gc

    static_gc = prepare_static_gc("Isolated full DiT replay")
    if static_gc is not None:
        atexit.register(static_gc.close)
    (out / "STATIC_GC.json").write_text(
        json.dumps(
            static_gc.receipt
            if static_gc
            else dict(enabled=False, automatic_gc_enabled=gc.isenabled())
        )
    )

    if args.queued_encoder:
        from mock_encoder import replay_queued_encoder

        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            cleanup_dist_env_and_memory,
        )

        try:
            with (
                torch.inference_mode(mode=args.execution_context == "inference"),
                torch.no_grad(),
                set_forward_context(current_timestep=0, attn_metadata=None),
            ):
                outputs = asyncio.run(
                    replay_queued_encoder(
                        args.capture,
                        session,
                        frozen,
                        new_sampler,
                        server.pipeline_config,
                    )
                )
            torch.cuda.synchronize()
            if len(outputs) != len(frozen):
                raise RuntimeError("Queued DiT replay changed the output chunk count")
            comparison = []
            for result, expected in zip(outputs, frozen):
                actual = result.visual
                wanted = expected["expected_visual"]
                comparison.append(
                    dict(
                        latent_start=result.chunk.start,
                        latent_end=result.chunk.end,
                        exact=bool(torch.equal(actual, wanted)),
                        max_abs=(actual.float() - wanted.float()).abs().max().item(),
                    )
                )
            torch.save(
                [result.visual.cpu() for result in outputs], out / "replay-latents.pt"
            )
            (out / "replay-comparison.json").write_text(
                json.dumps(comparison, indent=2)
            )
            if not all(row["exact"] for row in comparison):
                raise RuntimeError("Queued DiT replay differs from captured outputs")
            (out / "COMPLETE").write_text("Queued Encoder-to-DiT replay passed\n")
        finally:
            cleanup_dist_env_and_memory()
        return

    @torch.inference_mode(mode=args.execution_context == "inference")
    @torch.no_grad()
    def run(capture=False, forward_probe=False, inputs_synchronized=False):
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

        before = model.register_forward_pre_hook(pre) if forward_probe else None
        after = model.register_forward_hook(post) if forward_probe else None
        if not inputs_synchronized:
            dist.barrier()
            torch.cuda.synchronize()
        input_ready_barrier_ns = time.perf_counter_ns()
        resident_allocated_bytes = torch.cuda.memory_allocated()
        resident_reserved_bytes = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        owner, lease = SimpleNamespace(), None

        def acquire(positive, ready):
            nonlocal lease
            request = ZingH3GenerationStage.kv_request(
                SimpleNamespace(
                    codecs=SimpleNamespace(spatial_ratio=geometry.spatial_ratio)
                ),
                sampler,
                session["inputs"],
                positive,
                source_shape=ready.source.shape,
                total_frames=None,
            )
            lease = pool.acquire(owner, request)
            return lease.kv_cache

        state = SourceReadySession(
            sampler,
            session["inputs"],
            seed=session["seed"],
            max_pending_chunks=2,
            max_pending_bytes=256 << 20,
            acquire_storage=acquire if pool is not None else None,
            defer_final=policy["defer_final"],
            prompt_joint=policy["prompt_joint"],
            initial_prompt_event_id=policy["initial_prompt_event_id"],
            initial_prompt_version=policy["initial_prompt_version"],
        )
        if capture and (args.capture / "rng").exists():
            state._stall_rng_probe_root = out / "rng"
        gc_observer = GCPauseObserver()
        values, events, host_rows = [], [], []
        try:
            with (
                gc_observer,
                set_forward_context(current_timestep=0, attn_metadata=None),
            ):
                for entry in frozen:
                    chunk = entry["chunk"]
                    a, b = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    prompt = prompts.get(entry.get("prompt_file"))
                    if prompt is not None and (
                        prompt.event_id,
                        prompt.prompt_version,
                    ) != (state.prompt.event_id, state.prompt.prompt_version):
                        state.update_prompt(
                            prompt.condition,
                            prompt.event_id,
                            prompt.prompt_version,
                            inputs=prompt.inputs,
                        )
                    history = (
                        []
                        if state.cache is None
                        else [c.block for c in state.cache.committed]
                    )
                    entered = time.perf_counter_ns()
                    a.record()
                    event_enqueued = time.perf_counter_ns()
                    state.append(
                        chunk.source,
                        chunk.noise,
                        provenance=chunk.provenance,
                        final=chunk.final,
                    )
                    if args.diagnostic_blocks is not None and not host_rows:

                        def fingerprint(tensor):
                            raw = (
                                tensor.detach()
                                .contiguous()
                                .reshape(-1)
                                .view(torch.uint8)
                                .cpu()
                                .numpy()
                                .tobytes()
                            )
                            return dict(
                                shape=list(tensor.shape),
                                dtype=str(tensor.dtype),
                                inference=torch.is_inference(tensor),
                                sha256=hashlib.sha256(raw).hexdigest(),
                            )

                        ready = state.pending[0]
                        (out / "EXECUTION_CONTEXT_ACTIVE.json").write_text(
                            json.dumps(
                                dict(
                                    inference_mode=torch.is_inference_mode_enabled(),
                                    grad_enabled=torch.is_grad_enabled(),
                                    source=fingerprint(ready.source),
                                    noise=fingerprint(ready.noise),
                                    prompt={
                                        key: fingerprint(tensor)
                                        for key, tensor in state.prompt.condition.items()
                                    },
                                    tf32=torch.backends.cuda.matmul.allow_tf32,
                                    reduced_precision_bf16=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                                    scope="Untimed numerical prefix diagnostic before first model step",
                                ),
                                indent=2,
                            )
                        )
                    # Production work_quantum=1 drains one ready chunk at a time.
                    results = list(state.drain_ready())
                    if len(results) != 1:
                        raise RuntimeError(
                            "One-chunk replay changed the production drain boundary"
                        )
                    result = results[0]
                    b.record()
                    returned = time.perf_counter_ns()
                    history_after = [c.block for c in state.cache.committed]
                    host_rows.append(
                        dict(
                            entered_ns=entered,
                            returned_ns=returned,
                            input_ready_barrier_ns=input_ready_barrier_ns,
                            cuda_event_enqueued_ns=event_enqueued,
                            latent_start=chunk.chunk.start,
                            latent_end=chunk.chunk.end,
                            block_index=chunk.chunk.chunk_id,
                            prompt_event_id=result.prompt_event_id,
                            source_shape=list(chunk.source.shape),
                            noise_shape=list(chunk.noise.shape),
                            ready_before_run=True,
                            session_complete_after=state.complete,
                            history_blocks_before=history,
                            history_blocks_after=history_after,
                            history_removed_blocks=sorted(
                                set(history) - set(history_after)
                            ),
                            initial_schedule=chunk.chunk.chunk_id
                            < policy["schedule"]["initial_block_count"],
                        )
                    )
                    events.append((a, b))
                    if capture:
                        # Warmup correctness readback is untimed; formal rounds do not copy outputs.
                        values.append(result.visual.detach().cpu())
                if not state.ended:
                    state.finish()
                assert state.complete
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            stats = dict(
                seconds=seconds,
                blocks=[
                    {**row, "cuda_stream_ms": a.elapsed_time(b)}
                    for row, (a, b) in zip(host_rows, events)
                ],
                block_timing_scope="Whole append+drain call and its asynchronous CUDA events; graph-internal phase/step timings are not independently measured",
                input_readiness_scope="All frozen input tensors transferred before the measured run; ready barrier is an upper bound, not an online encoder enqueue timestamp",
                kv_history_scope="Before/after committed block identities and removals, including window eviction, prompt reset and EOS disposal; allocator reset/compaction internals are not individually timed",
                block_ms=[a.elapsed_time(b) for a, b in events],
                transformer_hook_ms=(
                    sum(a.elapsed_time(b) for a, b in forward_events)
                    if forward_probe
                    else None
                ),
                transformer_hook_scope="Python forwards only; CUDA Graph replay may bypass hooks",
                forward_count=len(forward_events) if forward_probe else None,
                resident_allocated_bytes=resident_allocated_bytes,
                resident_reserved_bytes=resident_reserved_bytes,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                gc_events=list(gc_observer.events),
                gc_policy="normal automatic GC; static heap startup policy in STATIC_GC.json",
                gc_scope="chunk loop and session finish; excludes state construction and disposal",
            )
            return stats, values
        finally:
            if before is not None:
                before.remove()
            if after is not None:
                after.remove()
            state.dispose()
            if lease is not None:
                lease.release()
            if pool is not None and pool.available != pool.size:
                raise RuntimeError("Replay leaked the production KV pool lease")

    restore_rng_probe = None
    if (args.capture / "rng").exists():
        from diagnostics.dit_replay.dit_rng_probe import install

        restore_rng_probe = install()
    try:
        warmup, actual = run(capture=True)
    finally:
        if restore_rng_probe is not None:
            restore_rng_probe()
    # Do not infer an SP tolerance. Keep raw outputs and descriptive errors;
    # strict replay status is separate from timing completeness.
    comparison = []
    valid = len(actual) == len(frozen) and all(
        got.shape == row["expected_visual"].shape
        and got.dtype == row["expected_visual"].dtype
        and bool(torch.isfinite(got).all())
        and bool(torch.isfinite(row["expected_visual"]).all())
        for got, row in zip(actual, frozen)
    )
    # Every rank makes the same fail/continue decision before any next model
    # collective. A mismatch on one rank cannot leave its peers in the soak.
    flags = torch.tensor([int(valid)], dtype=torch.int64, device="cuda")
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if not bool(flags.item()):
        raise RuntimeError("DiT replay produced invalid or incomplete output")
    for got, row in zip(actual, frozen, strict=True):
        expected = row["expected_visual"]
        delta = got.float() - expected.float()
        comparison.append(
            dict(
                exact=bool(torch.equal(got, expected)), max_abs=delta.abs().max().item()
            )
        )
    torch.save([v.cpu() for v in actual], out / "replay-latents.pt")
    (out / "replay-comparison.json").write_text(json.dumps(comparison))
    flags.fill_(int(all(row["exact"] for row in comparison)))
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    strict_replay_passed = bool(flags.item())
    if args.diagnostic_blocks is not None:
        (out / "DIAGNOSTIC_ONLY.json").write_text(
            json.dumps(
                dict(
                    execution_context=args.execution_context,
                    captured_chunks=count,
                    replayed_chunks=len(comparison),
                    strict_prefix_equal=all(row["exact"] for row in comparison),
                    full_case_accepted=False,
                    formal_timing_collected=False,
                    scope="Supplemental prefix numerical diagnosis; no full-context or soak acceptance",
                ),
                indent=2,
            )
        )
        return
    decision = replay_collection_decision(
        valid=True,
        exact=strict_replay_passed,
        minimum_seconds=args.minimum_round_seconds,
        collect_mismatch=args.collect_soak_on_numerical_mismatch,
    )
    (out / "NUMERICAL_STATUS.json").write_text(json.dumps(decision, indent=2))
    del actual
    runs = []
    for round_index in range(args.measured_runs):
        if args.minimum_round_seconds:

            def prepare_session():
                # Rank skew from receipt writes must not satisfy the work gate.
                dist.barrier()
                torch.cuda.synchronize()

            def complete_session():
                session_stats, _ = run(inputs_synchronized=True)
                # Synchronize only after complete EOS/disposal, never per block.
                torch.cuda.synchronize()
                return dict(
                    input_frames=capture_metadata["input_frames"],
                    fully_replayed=True,
                    measurement=session_stats,
                    after_cleanup_allocated_bytes=torch.cuda.memory_allocated(),
                    after_cleanup_reserved_bytes=torch.cuda.memory_reserved(),
                )

            with GCPauseObserver() as resident_gc:
                stats = run_resident_round(
                    complete_session,
                    args.minimum_round_seconds,
                    prepare_session=prepare_session,
                    on_progress=lambda value: (
                        out / f"resident-round-{round_index}-progress.json"
                    ).write_text(json.dumps(value, indent=2)),
                    all_ranks_complete=lambda seconds: all_ranks_reached_duration(
                        seconds, args.minimum_round_seconds, device="cuda"
                    ),
                )
            stats["gc_events"] = resident_gc.events
            for event in stats["gc_events"]:
                event["overlaps_session_work"] = any(
                    event["start_ns"] < row["resident_session_returned_ns"]
                    and event["end_ns"] > row["resident_session_entered_ns"]
                    for row in stats["sessions"]
                )
            stats["gc_scope"] = (
                "Entire resident round including session lifecycle, receipt writes and coordination; outside-work events are marked; nested session GC records are not additional events"
            )
        else:
            stats, _ = run()
        runs.append(stats)
        (out / "measured.json").write_text(json.dumps(runs, indent=2))
    profile = None
    if args.profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            profile, _ = run(forward_probe=True)
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
                profile_seconds_separate=profile["seconds"] if profile else None,
                replay_policy=policy,
                execution_context=args.execution_context,
                timing_scope="Full model, frozen ready input, work_quantum=1; asynchronous CUDA events in formal rounds",
                measurement_mode=(
                    "diagnostic_resident_soak"
                    if decision["measurement_scope"] == "diagnostic_resident_soak"
                    else "resident_complete_case_soak"
                    if args.minimum_round_seconds
                    else "single_complete_context"
                ),
                minimum_round_seconds=args.minimum_round_seconds,
                **decision,
                collection_completed=True,
                stable=(
                    (max(seconds) - min(seconds)) / statistics.median(seconds) <= 0.1
                    if not args.minimum_round_seconds
                    else None
                ),
            ),
            indent=2,
        )
    )
    if args.collect_soak_on_numerical_mismatch and not strict_replay_passed:
        (out / "COLLECTION_COMPLETE.json").write_text(
            json.dumps(dict(collection_completed=True, **decision), indent=2)
        )
    else:
        (out / "COMPLETE").write_text(
            "Timing and raw replay outputs collected; inspect numerical status\n"
        )


if __name__ == "__main__":
    main()
