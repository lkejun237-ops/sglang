# SPDX-License-Identifier: Apache-2.0
"""Independent DiT worker using the original transformer and SP implementation.

For SP2, launch with torchrun --standalone --nproc-per-node=2 and --sp-degree=2.
Only rank 0 serves HTTP and owns external Encoder/Decoder connections.
"""

import argparse
import asyncio
import json
import os
import tempfile
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from .incremental import SourceReadySession
from .packing import H3InferenceInputs
from .prompt_update import admit_prompt_update, coordinate_prompt_admission
from .queued_dit import QueuedDiTSession
from .source_bridge import SourceChunkBridge
from .streaming_noise import STREAMING_NOISE_VERSION


def create_link_session(sampler, codecs, init, output, *, stage=None):
    from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
        Provenance,
        SourceSpan,
    )

    if (
        type(init.get("seed")) is not int
        or init["seed"] < 0
        or any(
            type(init.get(k)) is not int or init[k] < 1
            for k in ("width", "height", "fps")
        )
        or init["fps"] != codecs.target_fps
    ):
        raise ValueError("Invalid DiT session geometry or seed")
    inputs = None
    lease = None
    conditions = {}
    kv_pool = None if stage is None else stage.kv_pool

    def acquire_storage(positive, ready):
        nonlocal lease
        request = stage.kv_request(
            sampler,
            worker.state.inputs,
            positive,
            source_shape=ready.source.shape,
            total_frames=None,
        )
        lease = kv_pool.acquire(worker, request)
        return lease.kv_cache

    def release_storage():
        nonlocal lease
        if stage is not None and stage.block0_cache is not None:
            stage.block0_cache.finish_fill()
        if lease is not None:
            lease.release()
            lease = None

    def provenance(pixels, dependency):
        spans = []
        for source_id, first, last in worker.source_records:
            start, end = max(first, pixels.start), min(last, pixels.end)
            if start < end:
                spans.append(SourceSpan(source_id, start, start, end - start))
        return Provenance(pixels, tuple(spans), dependency)

    bridge = SourceChunkBridge(
        sampler.processor.chunk_plan,
        clip_frames=codecs.vae_frames_per_chunk,
        clip_latents=codecs.vae_latents_per_chunk,
        temporal_ratio=codecs.temporal_ratio,
        capacity_latents=max(sampler.processor.chunk_plan.lengths)
        + codecs.vae_latents_per_chunk,
        capacity_bytes=256 << 20,
        seed=init["seed"],
        provenance_for=provenance,
        target_shape=(
            sampler.processor.visual_channels,
            init["height"] // codecs.spatial_ratio,
            init["width"] // codecs.spatial_ratio,
        ),
    )

    def prepare(message):
        nonlocal inputs
        active = inputs if message["initial"] else worker.state.inputs
        candidate = active
        if "inputs" in message:
            candidate = H3InferenceInputs(
                **message["inputs"], positive=message["tensors"], source_visual=None
            )
            candidate.visual_shape = tuple(candidate.visual_shape)
            if candidate.audio_shape is not None:
                candidate.audio_shape = tuple(candidate.audio_shape)
            if (
                candidate.visual_shape[1] * codecs.spatial_ratio != init["height"]
                or candidate.visual_shape[2] * codecs.spatial_ratio != init["width"]
            ):
                raise ValueError("Prepared inputs do not match the session geometry")
            if not message["initial"]:
                # References can replace the prefix, never the output geometry.
                for name in (
                    "output_type",
                    "visual_shape",
                    "audio_shape",
                    "pixel_frame_count",
                    "sample_count",
                    "sample_rate",
                    "max_area_config",
                ):
                    if getattr(candidate, name) != getattr(active, name):
                        raise ValueError(
                            "Reference updates cannot replace session geometry"
                        )
        if candidate is None:
            raise ValueError("Initial conditioning requires prepared inputs")
        condition = message["tensors"]
        # Use the original admission/packing rules with the candidate references.
        if kv_pool is not None:
            coordinate_prompt_admission(
                lambda: admit_prompt_update(
                    stage,
                    sampler,
                    candidate,
                    condition,
                    source_shape=tuple(
                        init.get("source_encoding", {}).get(k, init[k])
                        // codecs.spatial_ratio
                        for k in ("height", "width")
                    ),
                )
            )
        else:
            _, height, width = candidate.visual_shape
            visual = condition["context"].new_zeros(
                (1, sampler.processor.visual_channels, 1, height, width)
            )
            sampler.processor.pack_inference_layout(candidate, condition, visual, None)
        if message.get("cache_key") is not None:
            conditions.clear()
            snapshot = copy(candidate)
            snapshot.positive = condition
            conditions[message.get("event_id")] = (message, snapshot)
        if message["initial"]:
            inputs = candidate
        return candidate if "inputs" in message else None

    def create(condition, event, version):
        inputs.positive = condition
        state = SourceReadySession(
            sampler,
            inputs,
            seed=init["seed"],
            max_pending_chunks=1,
            max_pending_bytes=256 << 20,
            initial_prompt_event_id=event,
            initial_prompt_version=version,
            acquire_storage=acquire_storage if kv_pool is not None else None,
        )

        cache = None if stage is None else stage.block0_cache
        if cache is not None:

            def select(event_id):
                item = conditions.get(event_id)
                if item is None:
                    return None
                message, candidate = item
                entry = cache.lookup(message["cache_key"])
                state.capture_initial_prefix = None
                if entry is None and message.get("cache_write", False):

                    def capture(prefix):
                        request = stage.kv_request(
                            sampler,
                            candidate,
                            sampler.processor.pack_inference_layout(
                                candidate,
                                candidate.positive,
                                candidate.positive["context"].new_zeros(
                                    (
                                        1,
                                        sampler.processor.visual_channels,
                                        1,
                                        *candidate.visual_shape[-2:],
                                    )
                                ),
                                None,
                            ),
                            source_shape=tuple(
                                init.get("source_encoding", {}).get(k, init[k])
                                // codecs.spatial_ratio
                                for k in ("height", "width")
                            ),
                            total_frames=None,
                        )
                        cache.fill(
                            message["cache_key"],
                            candidate,
                            message.get("conditioning_identity"),
                            request,
                            state.cache,
                        )
                        cache.register(
                            message["cache_key"],
                            game_id=message.get("game_id"),
                            group_id=message.get("group_id"),
                        )

                    state.capture_initial_prefix = capture
                if entry is None:
                    return None

                def restore(target):
                    cache.restore(target, entry).commit(target)

                return restore

            startup = select(event)
            if startup is not None:
                state.restore_initial_prefix = startup

                def load(target, candidate, prompt, prefix):
                    if state.applied_prompt is not None:
                        return False
                    startup(target)
                    return True

                if sampler.joint_forward == "cross_chunk":
                    state.prefix_loader = load
            state.prepare_prompt_prefix = lambda candidate: select(candidate.event_id)
        return state

    worker = QueuedDiTSession(
        session_id=init["session_id"],
        generation_id=init["generation_id"],
        bridge=bridge,
        create_session=create,
        prepare_condition=prepare,
        release_storage=release_storage,
        on_output=output,
        sample_moments=lambda value, latent_start: (
            codecs.sample_streaming_visual_moments(
                value, latent_start=latent_start, noise_version=STREAMING_NOISE_VERSION
            )
        ),
    )
    worker.source_records = []
    return worker


def main():
    from sglang.multimodal_gen.configs.zing_sampling import parse_noise_grid

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sp-degree", type=int, choices=(1, 2), default=1)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, default=34000)
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--advertise-host", default="127.0.0.1")
    parser.add_argument("--rendezvous-port", type=int, default=35000)
    parser.add_argument("--worker-id")
    parser.add_argument("--service-count", type=int, choices=(1,), default=1)
    parser.add_argument(
        "--http-port", type=int, default=33000, help="Serve worker health on this port"
    )
    parser.add_argument("--pipeline-config-path", type=Path)
    parser.add_argument("--zing-solver", choices=("euler", "few_step"))
    parser.add_argument("--zing-denoising-step-list", type=parse_noise_grid)
    parser.add_argument("--zing-initial-denoising-step-list", type=parse_noise_grid)
    parser.add_argument("--zing-initial-block-count", type=int, choices=(0, 3))
    parser.add_argument(
        "--zing-source-joint-forward",
        choices=("off", "source_first", "cross_chunk"),
        help="Override the deployment profile's joint-forward policy",
    )
    parser.add_argument(
        "--quantization",
        help="Use the original transformer loader's quantization method (e.g. fp8)",
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--zing-block0-cache-mode", choices=("on", "off"), default="off"
    )
    parser.add_argument("--zing-block0-cache-host-gb", type=float)
    parser.add_argument("--zing-block0-cache-dir")
    parser.add_argument("--zing-block0-cache-disk-gb", type=float)
    parser.add_argument("--zing-block0-cache-gc", action="store_true")
    parser.add_argument("--zing-block0-cache-gc-interval-s", type=float, default=60.0)
    parser.add_argument("--zing-block0-cache-gc-stale-s", type=float, default=600.0)
    parser.add_argument("--zing-block0-cache-gc-low-watermark", type=float, default=0.9)
    cli = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    leader = int(os.environ.get("RANK", "0")) == 0
    if cli.sp_degree < 1 or cli.sp_degree != world_size:
        parser.error("--sp-degree must equal WORLD_SIZE; use torchrun for SP > 1")

    from .encoder_worker import initialize_runtime_imports

    ServerArgs, set_global_server_args = initialize_runtime_imports()
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        get_world_group,
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

    from .stage import ZingH3GenerationStage

    args = ServerArgs.from_kwargs(
        model_path=str(cli.model),
        pipeline_config_path=str(cli.pipeline_config_path)
        if cli.pipeline_config_path
        else None,
        num_gpus=world_size,
        quantization=cli.quantization,
        sp_degree=cli.sp_degree,
        ulysses_degree=cli.sp_degree,
        ring_degree=1,
        encoder_parallel="replicate",
        attention_backend="torch_sdpa",
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        use_fsdp_inference=False,
        performance_mode="speed",
        warmup_mode="off",
        realtime_max_sessions=1,
        realtime_max_sessions_per_worker=1,
        zing_source_joint_forward=cli.zing_source_joint_forward,
        **{
            name: value
            for name, value in vars(cli).items()
            if name.startswith("zing_block0_cache_")
        },
        # Preserve explicit CLI/profile precedence from the original entrypoint.
        **{
            name: getattr(cli, name)
            for name in (
                "zing_solver",
                "zing_denoising_step_list",
                "zing_initial_denoising_step_list",
                "zing_initial_block_count",
            )
            if getattr(cli, name) is not None
        },
    )
    set_global_server_args(args)
    link = None
    with tempfile.TemporaryDirectory(prefix="zing-dit-tp1-") as rendezvous:
        http = None
        try:
            if leader and cli.http_port is not None:
                from sglang.multimodal_gen.runtime.realtime.worker_http import (
                    WorkerHTTPServer,
                )

                http = WorkerHTTPServer("dit", cli.http_host, cli.http_port)
            torch.cuda.set_device(local_rank)
            maybe_init_distributed_environment_and_model_parallel(
                tp_size=1,
                sp_size=cli.sp_degree,
                ulysses_degree=cli.sp_degree,
                distributed_init_method=(
                    "env://"
                    if "RANK" in os.environ
                    else "file://" + rendezvous + "/group"
                ),
            )
            model, _ = TransformerLoader().load(
                str(cli.model / "transformer"),
                args,
                "transformer",
                "diffusers",
                allow_native_fallback=False,
            )
            model.eval()
            runtime = json.loads((cli.model / "zing_config.json").read_text())
            profile = json.loads((cli.model / "zing_realtime_profile.json").read_text())
            profile = _with_source_pipeline_policy(
                profile, joint_forward=args.zing_source_joint_forward
            )
            # The DiT only needs the original VAE geometry/posterior metadata.
            # Do not instantiate or load a VAE on this worker.
            args.pipeline_config.vae_config.update_model_arch(
                json.loads((cli.model / "video_vae/config.json").read_text())
            )
            stage = ZingH3GenerationStage(
                {"transformer": model},
                args.pipeline_config,
                runtime,
                external_inputs=True,
            )
            stage.model_path = str(cli.model)
            codecs = stage.codecs
            schedule = _source_sampling_schedule(args, profile)
            window = profile.get(
                "source_attention_window", {"window_chunks": 21, "sink_chunks": 3}
            )
            params = SimpleNamespace(
                num_inference_steps=len(schedule["denoising_step_list"]),
                guidance_scale=profile["guidance_scale"],
                # Match the original source adapter's deployment-profile policy.
                zing_joint_forward=profile.get("joint_forward", "cross_chunk"),
                zing_window_chunks=window["window_chunks"],
                zing_sink_chunks=window["sink_chunks"],
                zing_window_eviction_timing=profile.get(
                    "window_eviction_timing", "after_step0"
                ),
            )
            stage.deployment_sampling = schedule
            print(
                "DIT_SAMPLING_POLICY "
                + json.dumps(
                    dict(
                        joint_forward=params.zing_joint_forward,
                        **{
                            key: schedule.get(key)
                            for key in (
                                "solver",
                                "denoising_step_list",
                                "initial_block_count",
                                "initial_denoising_step_list",
                            )
                        },
                    )
                ),
                flush=True,
            )
            # Share the original stage initialization before accepting input.
            # The configured pool must be used by every linked session below.
            stage._build_realtime_sampler(params)
            stage.initialize_realtime_kv_cache_pool(args)

            def create(init, output):
                sampler = stage._build_realtime_sampler(params)
                return create_link_session(sampler, codecs, init, output, stage=stage)

            from .dit_preparation import DiTPreparation

            preparation = DiTPreparation(
                stage, lambda: stage._build_realtime_sampler(params)
            )
            create.cache_control = preparation.control
            create.cached_condition = preparation.cached_condition

            import uuid

            from sglang.multimodal_gen.runtime.realtime.worker_connection_ports import (
                ConnectionPorts,
            )
            from sglang.multimodal_gen.runtime.realtime.worker_group_dit import (
                DiTGroups,
            )

            parallel = get_world_group() if world_size > 1 else None
            ports = None
            descriptor = None
            try:
                if leader:
                    ports = ConnectionPorts(
                        "dit",
                        1,
                        worker_id=cli.worker_id or "dit-" + uuid.uuid4().hex,
                        host=cli.advertise_host,
                        bind_host=cli.http_host,
                        http=cli.http_port,
                        zmq=cli.port,
                        rendezvous=cli.rendezvous_port,
                    )
                    ports.initialize_stores(timeout=cli.timeout_seconds)
                    descriptor = ports.services[0]
                if parallel is not None:
                    descriptor = parallel.broadcast_object(descriptor)
                groups = DiTGroups(
                    descriptor,
                    store=ports.stores[0] if leader else None,
                    device=f"cuda:{local_rank}",
                    parallel=parallel,
                    timeout=cli.timeout_seconds,
                    create_session=create,
                    codecs=codecs,
                )
                if leader:
                    http.connections = groups.manager
                    groups.manager.install(http.app)
                    http.ready = True
                    print("DIT_SERVICE_READY " + json.dumps(descriptor), flush=True)
            except BaseException:
                if ports is not None:
                    ports.close()
                raise
            try:
                with (
                    torch.inference_mode(),
                    set_forward_context(current_timestep=0, attn_metadata=None),
                ):
                    asyncio.run(groups.run())
            finally:
                if ports is not None:
                    ports.close()

        finally:
            if http is not None:
                http.close()
            if link is not None:
                link.close()
            cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
