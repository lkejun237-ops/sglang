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
import traceback
import uuid
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


def _initialize_service_descriptor(ports, parallel, *, timeout, startup_failure=None):
    """Publish startup failure before peers can wait for a descriptor forever."""
    descriptor = None
    failure = startup_failure
    if ports is not None and failure is None:
        try:
            ports.initialize_stores(timeout=timeout)
            descriptor = ports.services[0]
        except BaseException as exc:
            traceback.print_exc()
            failure = exc
    result = (descriptor, failure is not None)
    if parallel is not None:
        result = parallel.broadcast_object(result)
    descriptor, failed = result
    if failed:
        if failure is not None:
            raise failure
        raise RuntimeError("DiT service initialization failed on the leader")
    return descriptor


def create_link_session(
    sampler, codecs, init, output, *, stage=None, defer_final=False, prompt_joint=False
):
    from sglang.multimodal_gen.runtime.realtime.zing_contracts import (
        Provenance,
        SourceSpan,
    )

    if type(init.get("detailed_timing", False)) is not bool:
        raise ValueError("Invalid detailed timing mode")
    if (
        type(defer_final) is not bool
        or type(prompt_joint) is not bool
        or type(init.get("seed")) is not int
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
    condition_keys = {}
    cache_reports = {}
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

    def prepare_candidate(message):
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
            proposed = copy(candidate)
            proposed.positive = condition
            stage.codecs.validate_prepared_compute_inputs(proposed)
            admit_prompt_update(
                stage,
                sampler,
                candidate,
                condition,
                source_shape=tuple(
                    init.get("source_encoding", {}).get(k, init[k])
                    // codecs.spatial_ratio
                    for k in ("height", "width")
                ),
                prompt_update=not message["initial"],
                prompt_joint=prompt_joint
                and (
                    message.get("source_boundary") is None
                    or message.get("allow_deferred_kv", False)
                ),
            )
        else:
            _, height, width = candidate.visual_shape
            visual = condition["context"].new_zeros(
                (1, sampler.processor.visual_channels, 1, height, width)
            )
            sampler.processor.pack_inference_layout(candidate, condition, visual, None)
        return candidate

    def prepare(message):
        nonlocal inputs
        # Include geometry and packing, with or without a KV pool, in the same
        # vote. Publish snapshots only after the leader accepts every rank.
        candidate = coordinate_prompt_admission(lambda: prepare_candidate(message))
        live = None if message["initial"] else worker.state.prompt
        key = message.get("cache_key")
        # This content key is generated by the original Encoder preparation;
        # it includes references and model identity, not just text/event IDs.
        reuse = (
            isinstance(key, str)
            and bool(key)
            and live is not None
            and key == condition_keys.get(live.event_id)
            and worker.state.applied_prompt is live
            and worker.state.cache is not None
            and worker.state.cache.prefix is not None
        )
        from sglang.multimodal_gen.runtime.conditioning.worker import (
            coordinate_validation,
        )

        votes = coordinate_validation((reuse, key))
        message["_reuse_gpu_prefix"] = reuse and all(
            v == votes[0] and v[0] for v in votes
        )
        keep = {message.get("event_id"), None if live is None else live.event_id}
        for old in list(condition_keys):
            if old not in keep:
                del condition_keys[old]
                cache_reports.pop(old, None)
        condition_keys[message.get("event_id")] = key
        condition = message["tensors"]
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
            metric_geometry={"width": init["width"], "height": init["height"]},
            seed=init["seed"],
            max_pending_chunks=1,
            max_pending_bytes=256 << 20,
            initial_prompt_event_id=event,
            initial_prompt_version=version,
            acquire_storage=acquire_storage if kv_pool is not None else None,
            defer_final=defer_final,
            prompt_joint=prompt_joint,
        )

        cache = None if stage is None else stage.block0_cache
        if cache is not None:

            def select(event_id):
                from .block0_telemetry import Block0Telemetry

                item = conditions.get(event_id)
                if item is None:
                    return None
                message, candidate = item
                telemetry = Block0Telemetry()
                entry = cache.lookup(message["cache_key"])
                lookup = dict(cache.last_lookup)
                earlier = message.get("cache_lookup", {})
                if (
                    lookup.get("cache_result") == "l1_hit"
                    and earlier.get("cache_result") == "l2_hit"
                ):
                    lookup = dict(earlier)
                telemetry.cache_result = lookup.get("cache_result", "miss")
                telemetry.miss_reason = lookup.get("miss_reason")
                telemetry.lookup_details = lookup
                if lookup.get("lookup_host_ms") is not None:
                    telemetry.record("cache_lookup", lookup["lookup_host_ms"])
                cache_reports[event_id] = dict(
                    cache_key=message["cache_key"],
                    prompt_version=message.get("prompt_version") or 1,
                    telemetry=telemetry,
                    restore=None,
                    reported=False,
                )
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
                        stored = cache.fill(
                            message["cache_key"],
                            candidate,
                            message.get("conditioning_identity"),
                            request,
                            state.cache,
                            required=False,
                        )
                        if stored:
                            cache.register(
                                message["cache_key"],
                                game_id=message.get("game_id"),
                                group_id=message.get("group_id"),
                            )

                    state.capture_initial_prefix = capture
                if entry is None:
                    return None

                def restore(target):
                    with telemetry.measure("restore"):
                        transfer = cache.restore(target, entry)
                        transfer.commit(target)
                    cache_reports[event_id]["restore"] = transfer

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

            def prepare_prefix(candidate):
                if not candidate.reuse_prefix:
                    return select(candidate.event_id)
                from .block0_telemetry import Block0Telemetry

                telemetry = Block0Telemetry()
                telemetry.cache_result = "bypass"
                telemetry.lookup_details = {"reuse_source": "resident_gpu_prefix"}
                telemetry.skip(
                    "cache_lookup",
                    "qwen_prepare",
                    "reference_vae",
                    "block0_forward",
                    "restore",
                )
                cache_reports[candidate.event_id] = dict(
                    cache_key=condition_keys[candidate.event_id],
                    prompt_version=candidate.prompt_version or 1,
                    telemetry=telemetry,
                    restore=None,
                    reported=False,
                )
                return None

            state.prepare_prompt_prefix = prepare_prefix
        return state

    def confirm_condition_ready(receipt):
        from sglang.multimodal_gen.runtime.conditioning.worker import (
            coordinate_validation,
        )

        # Match the integrated condition barrier: all ranks finish their CUDA
        # dependencies before publishing the same event/version/boundary.
        device = worker.state.device
        if device is None:
            # Initial conditioning can finish before the first source frame.
            device = next(sampler.generator.parameters()).device
        if device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            event.synchronize()
        if any(value != receipt for value in coordinate_validation(receipt)):
            raise ValueError("Condition readiness differs across DiT ranks")

    worker = QueuedDiTSession(
        detailed_timing=init.get("detailed_timing", False),
        session_id=init["session_id"],
        generation_id=init["generation_id"],
        bridge=bridge,
        return_stage_timing=bool(init.get("return_stage_timing", False)),
        create_session=create,
        prepare_condition=prepare,
        confirm_condition_ready=confirm_condition_ready,
        release_storage=release_storage,
        on_output=output,
        sample_moments=lambda value, latent_start: (
            codecs.sample_streaming_visual_moments(
                value, latent_start=latent_start, noise_version=STREAMING_NOISE_VERSION
            )
        ),
    )
    worker.source_records = []

    def take_cache_report(event_id):
        value = cache_reports.get(event_id)
        if value is None or value["reported"]:
            return None
        transfer, telemetry = value["restore"], value["telemetry"]
        if transfer is not None:
            # Match the integrated telemetry: never synchronize for reporting.
            if not transfer.end.query():
                return None
            telemetry.cuda["restore"] = transfer.begin.elapsed_time(transfer.end)
        telemetry.finish()
        value["reported"] = True
        report = dict(
            cache_key=value["cache_key"],
            prompt_version=value["prompt_version"],
            **telemetry.snapshot(),
        )
        print(
            "DIT_CACHE_RECEIPT "
            + json.dumps(
                dict(
                    session_id=init["session_id"],
                    generation_id=init["generation_id"],
                    rank=torch.distributed.get_rank()
                    if torch.distributed.is_initialized()
                    else 0,
                    **report,
                )
            ),
            flush=True,
        )
        return report

    worker.take_cache_report = take_cache_report
    return worker


def argument_parser():
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
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        help="Fraction of available GPU memory budgeted for resident models and KV",
    )
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
        "--zing-source-prompt-joint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Join prompt history with the next source/target step; overrides SGLANG_ZING_SOURCE_PROMPT_JOINT",
    )
    parser.add_argument(
        "--zing-source-defer-final",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defer single-chunk final KV commit into the next chunk's resume step",
    )
    parser.add_argument(
        "--quantization",
        help="Use the original transformer loader's quantization method (e.g. fp8)",
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--zing-block0-cache-mode", choices=("on", "off"), default="off"
    )
    parser.add_argument(
        "--zing-block0-cache-backend", choices=("local", "mooncake"), default="local"
    )
    parser.add_argument("--zing-block0-mooncake-config")
    parser.add_argument("--zing-block0-cache-host-gb", type=float)
    parser.add_argument("--zing-block0-cache-dir")
    parser.add_argument(
        "--zing-block0-cache-storage",
        choices=("mounted", "local"),
        default="mounted",
        help="Require a mounted cache volume, or explicitly use a job-local disk",
    )
    parser.add_argument("--zing-block0-cache-disk-gb", type=float)
    parser.add_argument("--zing-block0-cache-budget-root")
    parser.add_argument("--zing-block0-cache-min-free-gb", type=float, default=0.0)
    parser.add_argument("--zing-block0-cache-inflight-gb", type=float, default=128.0)
    parser.add_argument("--zing-block0-cache-max-writers", type=int, default=2)
    parser.add_argument(
        "--zing-block0-cache-gc-high-watermark", type=float, default=1.0
    )
    parser.add_argument("--zing-block0-cache-capacity-dir")
    parser.add_argument("--zing-block0-cache-host-total-gb", type=float)
    parser.add_argument("--zing-block0-cache-gc", action="store_true")
    parser.add_argument("--zing-block0-cache-gc-interval-s", type=float, default=60.0)
    parser.add_argument("--zing-block0-cache-gc-stale-s", type=float, default=600.0)
    parser.add_argument("--zing-block0-cache-gc-low-watermark", type=float, default=0.9)
    return parser


def prompt_joint_policy(cli):
    """Resolve at the process boundary; sessions receive an explicit boolean."""
    if cli.zing_source_prompt_joint is not None:
        return cli.zing_source_prompt_joint
    from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var

    return get_bool_env_var("SGLANG_ZING_SOURCE_PROMPT_JOINT", "false")


def main():
    parser = argument_parser()
    cli = parser.parse_args()
    from sglang.multimodal_gen.runtime.realtime.zing_nccl import configure_zing_nccl

    configure_zing_nccl()

    from sglang.multimodal_gen.runtime.realtime.role_pool.reporting import (
        validate_worker_configuration,
    )

    validate_worker_configuration("dit", cli.service_count)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    leader = int(os.environ.get("RANK", "0")) == 0
    if cli.sp_degree < 1 or cli.sp_degree != world_size:
        parser.error("--sp-degree must equal WORLD_SIZE; use torchrun for SP > 1")

    from sglang.multimodal_gen.runtime.realtime.zing_nccl import configure_zing_nccl

    configure_zing_nccl()

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
        pipeline_config_path=(
            str(cli.pipeline_config_path) if cli.pipeline_config_path else None
        ),
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
        mem_fraction_static=cli.mem_fraction_static,
        realtime_max_sessions=1,
        realtime_max_sessions_per_worker=1,
        zing_source_joint_forward=cli.zing_source_joint_forward,
        **{
            name: value
            for name, value in vars(cli).items()
            if name.startswith("zing_block0_cache_")
            or name == "zing_block0_mooncake_config"
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
    static_gc = None
    with tempfile.TemporaryDirectory(prefix="zing-dit-tp1-") as rendezvous:
        http = None
        ports = None
        try:
            startup_failure = None
            try:
                if leader and cli.http_port is not None:
                    from sglang.multimodal_gen.runtime.realtime.worker_http import (
                        WorkerHTTPServer,
                    )

                    http = WorkerHTTPServer("dit", cli.http_host, cli.http_port)
                if leader:
                    from sglang.multimodal_gen.runtime.realtime.worker_connection_ports import (
                        ConnectionPorts,
                    )

                    # Reserve fixed service ports before Gloo/NCCL choose ephemeral
                    # sockets. Otherwise those sockets can occupy our own endpoint.
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
            except BaseException as exc:
                # Keep the leader alive until every SP rank can observe failure.
                traceback.print_exc()
                startup_failure = exc
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
            parallel = get_world_group() if world_size > 1 else None
            descriptor = _initialize_service_descriptor(
                ports,
                parallel,
                timeout=cli.timeout_seconds,
                startup_failure=startup_failure,
            )
            before_load = None
            if args.mem_fraction_static is not None:
                from sglang.multimodal_gen.runtime.realtime.zing_kv_pool import (
                    cleaned_free_memory,
                )

                before_load = cleaned_free_memory(torch.device("cuda", local_rank))
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
                "source_attention_window", {"window_chunks": 21, "sink_chunks": 2}
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
            prompt_joint = prompt_joint_policy(cli)
            if cli.zing_source_defer_final is not None:
                defer_final = cli.zing_source_defer_final
            elif "SGLANG_ZING_SOURCE_DEFER_FINAL" in os.environ:
                defer_final = os.environ["SGLANG_ZING_SOURCE_DEFER_FINAL"] == "1"
            else:
                defer_final = (
                    params.zing_joint_forward == "cross_chunk"
                    and params.zing_window_eviction_timing == "after_step0"
                    and float(params.guidance_scale) == 1.0
                )
            print(
                "DIT_SAMPLING_POLICY "
                + json.dumps(
                    dict(
                        joint_forward=params.zing_joint_forward,
                        source_defer_final=defer_final,
                        source_prompt_joint=prompt_joint,
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
            stage.initialize_realtime_kv_cache_pool(args, before_load=before_load)

            def create(init, output):
                sampler = stage._build_realtime_sampler(params)
                return create_link_session(
                    sampler,
                    codecs,
                    init,
                    output,
                    stage=stage,
                    defer_final=defer_final,
                    prompt_joint=prompt_joint,
                )

            from .dit_preparation import DiTPreparation

            preparation = DiTPreparation(
                stage, lambda: stage._build_realtime_sampler(params)
            )
            create.cache_control = preparation.control
            create.cached_condition = preparation.cached_condition

            from .dit_warmup import warmup_dit

            warmup_dit(model, codecs, create, parallel=parallel)

            from sglang.multimodal_gen.runtime.realtime.worker_group_dit import (
                DiTGroups,
            )

            groups = DiTGroups(
                descriptor,
                store=ports.stores[0] if leader else None,
                device=f"cuda:{local_rank}",
                parallel=parallel,
                timeout=cli.timeout_seconds,
                create_session=create,
                codecs=codecs,
            )
            from sglang.multimodal_gen.runtime.utils.zing_static_gc import (
                prepare_static_gc,
            )

            if leader:
                from sglang.multimodal_gen.runtime.realtime.prefill_persistence import (
                    inspect_block0_persistence,
                )

                def persistence(descriptor):
                    return inspect_block0_persistence(stage.block0_cache, descriptor)

                groups.manager.persistence = persistence

            static_gc = prepare_static_gc("Zing standalone DiT worker")
            if leader:
                http.connections = groups.manager
                groups.manager.install(http.app)
                http.ready = True
                print("DIT_SERVICE_READY " + json.dumps(descriptor), flush=True)
            with (
                torch.inference_mode(),
                set_forward_context(current_timestep=0, attn_metadata=None),
            ):
                asyncio.run(groups.run())
        except BaseException:
            # A failed cleanup must not hide the original startup/model error.
            traceback.print_exc()
            raise
        finally:
            if ports is not None:
                ports.close()
            if http is not None:
                http.close()
            if link is not None:
                link.close()
            if static_gc is not None:
                static_gc.close()
            cleanup_dist_env_and_memory()


if __name__ == "__main__":
    from sglang.multimodal_gen.runtime.realtime.worker_process import run_worker

    run_worker(main)
