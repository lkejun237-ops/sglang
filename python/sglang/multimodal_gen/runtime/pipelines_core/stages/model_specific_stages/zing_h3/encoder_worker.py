# SPDX-License-Identifier: Apache-2.0
"""Single-session, same-node encoder service, launched with torchrun.

Bind only loopback. CUDA_VISIBLE_DEVICES must contain only the dedicated encoder
GPUs. This is an experimental deployment component, not a public network API.
"""

import argparse
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import zmq

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
    args = parser.parse_args()
    ServerArgs, set_global_server_args = initialize_runtime_imports()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout_seconds))
    rank = dist.get_rank()
    control_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=30))
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.parallel_encoder import (
        create_encoder_cache,
    )

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
    socket = None
    if rank == 0:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(f"tcp://127.0.0.1:{args.port}")
    dist.barrier()
    print(
        json.dumps(
            {
                "encoder_service_ready": True,
                "rank": rank,
                "world": dist.get_world_size(),
            }
        ),
        flush=True,
    )
    session = bridge = token = profiler = None
    last_released = None
    quarantined = False
    try:
        while True:
            command, payload = None, None
            if rank == 0:
                parts = socket.recv_multipart()
                command = json.loads(parts[0])
                payload = parts[1] if len(parts) > 1 else None
            commands = [command]
            dist.broadcast_object_list(commands, src=0)
            command = commands[0]
            op = command["op"]
            started = time.time_ns()
            if quarantined and op not in ("ping", "abort"):
                if rank == 0:
                    socket.send_json(
                        dict(
                            ok=False,
                            code="ENCODER_RESTART_REQUIRED",
                            error="The encoder must be restarted before another operation.",
                        )
                    )
                continue
            if op == "ping":
                reply = {
                    "ok": True,
                    "world_size": dist.get_world_size(),
                    "busy": session is not None,
                    "restart_required": quarantined,
                }
            elif op == "abort":

                def consensus(value):
                    values = [None] * dist.get_world_size()
                    dist.all_gather_object(values, value, group=control_group)
                    return values

                def cleanup_owner():
                    nonlocal session, bridge, profiler
                    if session is not None:
                        session.dispose()
                    if profiler is not None:
                        profiler.stop()
                        profiler = None
                    bridge = None

                reply, quarantine = abort_owner(
                    command, token, last_released, quarantined, cleanup_owner, consensus
                )
                quarantined = quarantined or quarantine
                if reply.get("released"):
                    last_released = token
                    session = bridge = token = None
            elif op == "open":
                if session is not None:
                    # Reject before changing the owner or joining another data
                    # bridge. Every rank takes this branch after the same control
                    # broadcast; the current session remains usable.
                    if rank == 0:
                        socket.send_json(
                            dict(
                                ok=False,
                                code="ENCODER_BUSY",
                                error="The encoder is busy. Please try again later.",
                            )
                        )
                    continue
                token = command["session"]
                profiler = None
                cache = create_encoder_cache(
                    codecs.modules["video_vae"],
                    dist.group.WORLD,
                    control_group=control_group,
                    encode_mode=codecs.runtime["vae_config"]["encode_mode"],
                    encode_dtype=codecs.runtime["vae_config"]["encode_dtype"],
                )
                session = H3ProgressiveEncoderSession(
                    codecs, capacity_frames=command["capacity_frames"], cache=cache
                )
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
                if session is None or command.get("session") != token:
                    raise ValueError("Encoder session does not match")
                value = None
                if op == "append":
                    shape = command["shape"]
                    if (
                        len(shape) != 4
                        or shape[0] != 3
                        or not 1 <= shape[1] <= 51
                        or min(shape) < 1
                    ):
                        raise ValueError("Invalid encoder input shape")
                    if math.prod(shape) > command["capacity_bytes"]:
                        raise ValueError("Encoder input exceeds byte capacity")
                    pixels = torch.empty(shape, dtype=torch.uint8, device="cuda")
                    if rank == 0:
                        pixels.copy_(
                            torch.frombuffer(
                                bytearray(payload), dtype=torch.uint8
                            ).reshape(shape)
                        )
                    with torch.profiler.record_function(
                        "Zing.encoder_input_distribution"
                    ):
                        dist.broadcast(pixels, src=0)
                    with torch.profiler.record_function(
                        f"Zing.encoder_admission.source_{command['frame_start']}_{command['frame_start'] + shape[1]}.dispatch_end_{command.get('dispatch_accepted_frames', -1)}"
                    ):
                        session.append(pixels, frame_start=command["frame_start"])
                elif op == "finish":
                    session.finish()
                elif op == "pop":
                    with torch.profiler.record_function(
                        f"Zing.clock_anchor.unix_ns{time.time_ns()}"
                    ):
                        pass
                    with torch.profiler.record_function(
                        f"Zing.encoder_prefetch.latent_start_{session.emitted_latents}.source_end_{command.get('prefetched_frames', -1)}.dispatch_end_{command.get('dispatch_accepted_frames', -1)}"
                    ):
                        value = session.pop()
                    if value is None:
                        raise ValueError("Encoder is not ready")
                elif op == "close":
                    session.dispose()
                else:
                    raise ValueError("Unknown encoder operation")
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
                if op == "close":
                    if profiler is not None:
                        profiler.stop()
                        directory = Path(command["profile_dir"])
                        directory.mkdir(parents=True, exist_ok=True)
                        profiler.export_chrome_trace(
                            str(directory / f"encoder-{token}-rank{rank}.trace.json")
                        )
                        profiler = None
                    # ACK means every rank finished cleanup/trace export. The
                    # harness may terminate this process group immediately after.
                    dist.barrier()
                if rank == 0:
                    socket.send_json(reply)
                    if value is not None:
                        with torch.profiler.record_function(
                            "Zing.encoder_output_transfer"
                        ):
                            bridge.send([value.values.contiguous()], 0, 0).wait()
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
                if op == "close":
                    last_released = token
                    session = bridge = token = None
                continue
            if rank == 0:
                socket.send_json(reply)
    finally:
        if session is not None:
            session.dispose()
        if socket is not None:
            socket.close()
            context.term()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
