# SPDX-License-Identifier: Apache-2.0
"""Single-session, same-node encoder service, launched with torchrun.

Bind only loopback. CUDA_VISIBLE_DEVICES must contain only the dedicated encoder
GPUs. This is an experimental deployment component, not a public network API.
"""

import argparse
import json
import logging
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import zmq

from sglang.multimodal_gen.runtime.utils.zing_stage_timing import (
    handoff,
    stage_span,
    trace_for_ids,
)

from .encoder_control import EncoderCommandControl, EncoderControlError
from .encoder_recovery import abort_owner
from .encoder_transport import bridge_group
from .progressive_encoder import H3ProgressiveEncoderSession


def initialize_runtime_imports():
    """Load pipeline registration before server arguments can import it cyclically.

    This bootstrap performs no CUDA initialization and can be checked in the
    preparation container before reserving GPUs. Never install platform stubs
    here: production must import its real GPU dependencies.
    """
    import sglang.multimodal_gen.configs.pipeline_configs  # noqa: F401
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )

    return ServerArgs, set_global_server_args


def load_encoder(root):
    from safetensors.torch import load_file

    from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
        MiniMaxH3VideoVAEConfig,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
        _consume_vae_checkpoint_arch_metadata,
    )
    from sglang.multimodal_gen.runtime.loader.utils import load_model_state_dict
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE

    from .codecs import ZingH3Codecs

    config = MiniMaxH3VideoVAEConfig()
    config.update_model_arch(json.loads((root / "video_vae/config.json").read_text()))
    model = MiniMaxH3VideoVAE(config)
    state = load_file(str(root / "video_vae/source/model.safetensors"))
    _consume_vae_checkpoint_arch_metadata(state, config, model.state_dict())
    load_model_state_dict(model, state, assign=True, strict=True)
    del state
    # Strict-load the original package, then drop the unused native decoder
    # before moving to CUDA. No decoder kernels run on these devices.
    del model.decoder
    if hasattr(model, "post_quant_conv"):
        del model.post_quant_conv
    codecs = object.__new__(ZingH3Codecs)
    codecs.modules = {"video_vae": model.cuda().eval()}
    codecs.va = config.arch_config
    codecs.temporal_ratio = codecs.va.temporal_compression_ratio
    codecs.vae_frames_per_chunk = codecs.va.vae_clip_length
    codecs.vae_latents_per_chunk = math.ceil(
        codecs.vae_frames_per_chunk / codecs.temporal_ratio
    )
    codecs.runtime = json.loads((root / "zing_config.json").read_text())
    codecs.prepare_visual_encoder_weights()
    return codecs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--startup-timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.startup_timeout_seconds <= 0:
        raise ValueError("Encoder timeouts must be positive")
    ServerArgs, set_global_server_args = initialize_runtime_imports()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout_seconds))
    rank = dist.get_rank()
    # No device-group collective is used while waiting for commands or weights.
    idle_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=30))
    control_group = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=args.timeout_seconds)
    )
    startup_group = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=args.startup_timeout_seconds)
    )
    control = EncoderCommandControl(
        dist,
        rank=rank,
        socket=None,
        idle_group=idle_group,
        operation_group=control_group,
    )
    session = bridge = token = profiler = socket = context = None
    stage_timing_dir = None
    last_released = None
    quarantined = False
    source_frames, source_shape, source_capacity, source_ended = 0, None, 0, False
    try:
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.parallel_encoder import (
            create_encoder_cache,
        )

        error = None
        try:
            set_global_server_args(
                ServerArgs.from_kwargs(
                    model_path=str(args.model),
                    num_gpus=1,
                    video_decoder_backend="native",
                    attention_backend="torch_sdpa",
                    warmup_mode="off",
                )
            )
            codecs = load_encoder(args.model)
            codecs.warmup_encoder()
            if rank == 0:
                context = zmq.Context()
                socket = context.socket(zmq.REP)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.SNDTIMEO, args.timeout_seconds * 1000)
                socket.bind(f"tcp://127.0.0.1:{args.port}")
                control.socket = socket
        except Exception as exc:
            logging.exception("Encoder startup failed on rank %s", rank)
            error = exc
        control.agree("startup", error=error, group=startup_group)
        print(
            json.dumps(
                {
                    "encoder_service_ready": True,
                    "rank": rank,
                    "world": control.world_size,
                }
            ),
            flush=True,
        )
        while True:
            command, payload = control.next()
            if command is None:
                continue
            op = command["op"]
            started = time.time_ns()
            # The full command, generation identity and local input cursor must
            # match before branching or touching a device input collective.
            control.agree(
                "command",
                (
                    control.sequence,
                    command,
                    token,
                    last_released,
                    quarantined,
                    session is not None,
                    source_frames,
                    source_shape,
                    source_capacity,
                    source_ended,
                ),
            )
            value = None
            error = None
            try:
                if quarantined and op not in ("ping", "abort"):
                    reply = dict(
                        ok=False,
                        code="ENCODER_RESTART_REQUIRED",
                        error="The encoder must be restarted before another operation.",
                    )
                elif op == "ping":
                    reply = dict(
                        ok=True,
                        world_size=control.world_size,
                        busy=session is not None,
                        restart_required=quarantined,
                    )
                elif op == "abort":

                    def cleanup_owner():
                        nonlocal session, bridge, profiler
                        if session is not None:
                            session.dispose()
                        if profiler is not None:
                            profiler.stop()
                            profiler = None
                        bridge = None

                    reply, quarantine = abort_owner(
                        command,
                        token,
                        last_released,
                        quarantined,
                        cleanup_owner,
                        control.gather,
                    )
                    quarantined = quarantined or quarantine
                    if reply.get("released"):
                        last_released = token
                        session = bridge = token = None
                        source_frames, source_shape, source_capacity, source_ended = (
                            0,
                            None,
                            0,
                            False,
                        )
                elif op == "open":
                    if session is not None:
                        reply = dict(
                            ok=False,
                            code="ENCODER_BUSY",
                            error="The encoder is busy. Please try again later.",
                        )
                    else:
                        error = None
                        try:
                            token = command["session"]
                            if not isinstance(token, str) or not token:
                                raise ValueError("Invalid encoder session")
                            stage_timing_dir = command.get("stage_timing_dir")
                            cache = create_encoder_cache(
                                codecs.modules["video_vae"],
                                dist.group.WORLD,
                                control_group=control_group,
                                encode_mode=codecs.runtime["vae_config"]["encode_mode"],
                                encode_dtype=codecs.runtime["vae_config"][
                                    "encode_dtype"
                                ],
                            )
                            source_capacity = command["capacity_frames"]
                            session = H3ProgressiveEncoderSession(
                                codecs, capacity_frames=source_capacity, cache=cache
                            )
                            source_frames, source_shape, source_ended = 0, None, False
                        except Exception as exc:
                            logging.exception(
                                "Encoder open preparation failed on rank %s", rank
                            )
                            error = exc
                        control.agree("open_prepared", token, error=error)
                        if rank == 0:
                            store = dist.TCPStore(
                                "127.0.0.1",
                                command["bridge_port"],
                                2,
                                False,
                                timeout=timedelta(seconds=args.timeout_seconds),
                            )
                            bridge = bridge_group(
                                store, 1, timeout_seconds=args.timeout_seconds
                            )
                        if command.get("profile_dir"):
                            profiler = torch.profiler.profile(
                                activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA,
                                ]
                            )
                            profiler.start()
                        reply = {"ok": True}
                else:
                    error = None
                    try:
                        if session is None or command.get("session") != token:
                            raise ValueError("Encoder session does not match")
                        if op not in ("append", "finish", "pop", "close"):
                            raise ValueError("Unknown encoder operation")
                        if op == "append":
                            shape = command["shape"]
                            if (
                                not isinstance(shape, list)
                                or len(shape) != 4
                                or any(type(n) is not int or n < 1 for n in shape)
                                or shape[0] != 3
                                or shape[1] > 51
                            ):
                                raise ValueError("Invalid encoder input shape")
                            if (
                                type(command["capacity_bytes"]) is not int
                                or math.prod(shape) > command["capacity_bytes"]
                            ):
                                raise ValueError("Encoder input exceeds byte capacity")
                            if (
                                source_ended
                                or type(command["frame_start"]) is not int
                                or command["frame_start"] != source_frames
                                or session.buffered_frames + shape[1] > source_capacity
                                or (
                                    source_shape is not None
                                    and shape[2:] != source_shape
                                )
                            ):
                                raise ValueError(
                                    "Encoder input does not match session state"
                                )
                            pixels = torch.empty(
                                shape, dtype=torch.uint8, device="cuda"
                            )
                            if rank == 0:
                                if payload is None or len(payload) != math.prod(shape):
                                    raise ValueError("Invalid encoder input payload")
                                pixels.copy_(
                                    torch.frombuffer(
                                        bytearray(payload), dtype=torch.uint8
                                    ).reshape(shape)
                                )
                            # Observe just this input's producer stream. A device
                            # synchronize would also stall unrelated model work.
                            input_ready = torch.cuda.Event()
                            input_ready.record(torch.cuda.current_stream())
                            input_ready.synchronize()
                        elif op == "pop" and not session.ready:
                            raise ValueError("Encoder is not ready")
                    except Exception as exc:
                        logging.exception(
                            "Encoder command preparation failed on rank %s", rank
                        )
                        error = exc
                    control.agree("prepared", (control.sequence, command), error=error)
                    if op == "append":
                        with torch.profiler.record_function(
                            "Zing.encoder_input_distribution"
                        ):
                            dist.broadcast(pixels, src=0)
                            input_distributed = torch.cuda.Event()
                            input_distributed.record(torch.cuda.current_stream())
                            input_distributed.synchronize()
                        with torch.profiler.record_function(
                            f"Zing.encoder_admission.source_{command['frame_start']}_{command['frame_start'] + shape[1]}.dispatch_end_{command.get('dispatch_accepted_frames', -1)}"
                        ):
                            session.append(pixels, frame_start=command["frame_start"])
                        source_frames += shape[1]
                        source_shape = shape[2:]
                    elif op == "finish":
                        session.finish()
                        source_ended = True
                    elif op == "pop":
                        with torch.profiler.record_function(
                            f"Zing.clock_anchor.unix_ns{time.time_ns()}"
                        ):
                            pass
                        latent_start = session.emitted_latents
                        trace = trace_for_ids(
                            token,
                            f"{token}:source:{latent_start}",
                            enabled=stage_timing_dir is not None,
                            directory=stage_timing_dir,
                            identity={
                                "worker_role": "dedicated_source_encoder",
                                "component_group": "encoder",
                            },
                        )
                        with stage_span(
                            trace,
                            "Encoder",
                            cuda=trace is not None and trace.measure_cuda,
                            component="source_vae",
                            input_ids=[
                                f"{token}:source:0:{command.get('prefetched_frames', -1)}"
                            ],
                            latent_start=latent_start,
                            dispatch_accepted_frames=command.get(
                                "dispatch_accepted_frames"
                            ),
                            prefetched_frames=command.get("prefetched_frames"),
                            scope="dedicated_source_VAE_encode_and_streaming_cache_update",
                        ) as span:
                            with torch.profiler.record_function(
                                f"Zing.encoder_prefetch.latent_start_{latent_start}.source_end_{command.get('prefetched_frames', -1)}.dispatch_end_{command.get('dispatch_accepted_frames', -1)}"
                            ):
                                value = session.pop()
                            if value is None:
                                raise ValueError("Encoder is not ready")
                        handoff(
                            span,
                            output_ids=[f"{token}:latent:{value.start}:{value.end}"],
                            latent_range=[value.start, value.end],
                            source_dependency_end=value.source_dependency_end,
                        )
                    elif op == "close":
                        session.dispose()
                        if profiler is not None:
                            profiler.stop()
                            directory = Path(command["profile_dir"])
                            directory.mkdir(parents=True, exist_ok=True)
                            profiler.export_chrome_trace(
                                str(
                                    directory / f"encoder-{token}-rank{rank}.trace.json"
                                )
                            )
                            profiler = None
                    reply = dict(
                        ok=True,
                        ready=session.ready,
                        state=vars(session.state_usage()),
                        buffered_frames=session.buffered_frames,
                        encoder_execution=session.cache.last_execution,
                    )
                    if value is not None:
                        reply.update(
                            dispatch_accepted_frames=command.get(
                                "dispatch_accepted_frames"
                            ),
                            prefetched_frames=command.get("prefetched_frames"),
                            shape=list(value.values.shape),
                            dtype=str(value.values.dtype).split(".")[-1],
                            start=value.start,
                            end=value.end,
                            dependency=value.source_dependency_end,
                        )
            except EncoderControlError:
                raise
            except Exception as exc:
                logging.exception("Encoder operation failed on rank %s", rank)
                error = exc
            # An execution failure ends the service. No peer may reuse a
            # possibly damaged communicator, even if its local work succeeded.
            control.agree("executed", control.sequence, error=error)
            error = None
            try:
                # Cleanup/encoding has finished on every rank before ACK.
                # Include the reply in the execution budget as well as tensors.
                control.reply(reply)
                if value is not None and rank == 0:
                    # The DiT receiver allocates from metadata before posting
                    # its receive. Sending the tensor first would deadlock.
                    with torch.profiler.record_function("Zing.encoder_output_transfer"):
                        bridge.send([value.values.contiguous()], 0, 0).wait()
                        output_sent = torch.cuda.Event()
                        output_sent.record(torch.cuda.current_stream())
                        output_sent.synchronize()
            except Exception as exc:
                logging.exception("Encoder output failed on rank %s", rank)
                error = exc
            # This uses the execution budget, never the idle heartbeat timeout.
            control.agree(f"{op}_complete", control.sequence, error=error)
            if op not in ("open", "abort", "ping") and reply.get("ok"):
                print(
                    json.dumps(
                        dict(
                            encoder_step=op,
                            session=token,
                            rank=rank,
                            host_start_ns=started,
                            host_end_ns=time.time_ns(),
                            gpu_allocated_bytes=torch.cuda.memory_allocated(),
                            gpu_reserved_bytes=torch.cuda.memory_reserved(),
                            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            **reply,
                        )
                    ),
                    flush=True,
                )
            if op == "close" and reply.get("ok"):
                last_released = token
                session = bridge = token = None
                source_frames, source_shape, source_capacity, source_ended = (
                    0,
                    None,
                    0,
                    False,
                )
    except Exception:
        logging.exception("Encoder service stopped on rank %s", rank)
        if rank == 0 and control.request_pending:
            try:
                control.reply(
                    dict(
                        ok=False,
                        code="ENCODER_RESTART_REQUIRED",
                        error="The encoder must be restarted before another operation.",
                    )
                )
            except Exception:
                logging.exception("Encoder failure reply could not be sent")
        raise
    finally:
        # Local tensor/cache disposal is safe; do not issue another collective
        # or a synchronizing NCCL shutdown after communication has failed.
        if session is not None:
            session.dispose()
        if socket is not None:
            socket.close()
        if context is not None:
            context.term()


if __name__ == "__main__":
    main()
