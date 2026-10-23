# SPDX-License-Identifier: Apache-2.0
"""Dedicated Lynn decoder service on its own GPU group.

The DiT leader sends normalized latents over a private NCCL bridge and receives
one shared-memory reference per dispatch instead of pixels. Decode, uint8
conversion, D2H and publication stay on the decoder host, so the DiT never
waits for decode. One active session; a failed generation ends the service
because Lynn K/V state is not retryable.
"""

import argparse
import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import zmq

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3_decoder import (
    normalize_decoder_spec,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    publish_async_shared_memory_error,
    publish_async_shared_memory_payload,
    reserve_async_shared_memory_payload,
    wait_for_async_shared_memory_terminal,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import stage_rgb24_segments

from .decoder_transport import bridge_group, receive_latent
from .encoder_control import EncoderCommandControl, EncoderControlError
from .source_decoder import check_latent_append

_MAX_OUTPUT_BYTES = 128 << 20
_DRAIN = object()


def initialize_runtime_imports():
    """Load pipeline registration before server arguments import it cyclically."""
    import sglang.multimodal_gen.configs.pipeline_configs  # noqa: F401
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )

    return ServerArgs, set_global_server_args


class _TileGroup:
    """The decoder world's SP interface, matching what Lynn decode_step uses."""

    def __init__(self, *, rank, world_size, cpu_group, device_group):
        self.rank_in_group = rank
        self.world_size = world_size
        self.cpu_group = cpu_group
        self.device_group = device_group

    def all_gather_into(self, output, tensor):
        dist.all_gather_into_tensor(output, tensor, group=self.device_group)


class _OutputPublisher:
    """Reserve a shared-memory reference per dispatch, publish after decode."""

    def __init__(self, *, timeout_seconds, max_in_flight=2):
        self.timeout = timeout_seconds
        self._slots = threading.BoundedSemaphore(max_in_flight)
        self._executor = ThreadPoolExecutor(
            max_workers=max_in_flight, thread_name_prefix="zing-decoder-publish"
        )

    def reserve(self, *, frames, width, height, identity):
        num_bytes = frames * width * height * 3
        if not 0 < num_bytes <= _MAX_OUTPUT_BYTES:
            raise ValueError("Decoder output exceeds its payload capacity")
        if not self._slots.acquire(timeout=self.timeout):
            raise TimeoutError("Decoder output slots are full")
        try:
            reference = reserve_async_shared_memory_payload(num_bytes)
            reference.update(
                kind="raw_rgb24_frames",
                version=1,
                batch_num_frames=[frames],
                width=width,
                height=height,
                channels=3,
                **identity,
            )
        except BaseException:
            self._slots.release()
            raise
        return reference

    def publish(self, reference, frames):
        device_frames, counts = stage_rgb24_segments([frames])
        if counts != reference["batch_num_frames"] or (
            device_frames.numel() != reference["num_bytes"]
        ):
            raise ValueError("Decoded RGB does not match the reserved reference")
        publish_async_shared_memory_payload(reference, device_frames.cpu().numpy())
        self._executor.submit(self._reclaim, reference)

    def fail(self, reference, error):
        try:
            publish_async_shared_memory_error(reference, error)
        except Exception:
            logging.exception("Decoder output error publication failed")
        self._executor.submit(self._reclaim, reference)

    def _reclaim(self, reference):
        try:
            wait_for_async_shared_memory_terminal(reference)
        except Exception:
            logging.exception("Decoder output reference was not reclaimed")
        finally:
            self._slots.release()

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=False)


class _DecodeService:
    """Input boundaries on the command thread, stream state on the decode thread."""

    def __init__(
        self, module, *, capacity_latents, group, publisher, decode_group, device
    ):
        self.protocol = module.protocol
        self.device = device
        self.spatial_ratio = module.spatial_ratio
        self._stream = module.new_stream(frame_limit=None, group=group)
        self.capacity_latents = capacity_latents
        self.publisher = publisher
        self.decode_group = decode_group
        self.failure = None
        self.width = self.height = None
        self.accepted_latents = self.submitted_latents = 0
        self.valid_frame_end = self.submitted_frames = 0
        self.total_frames = None
        self._pending = []
        self._held = None
        self._jobs = queue.Queue(maxsize=2)
        self._last_job = None
        self._tile_parallel = group is not None
        self._thread = threading.Thread(
            target=self._decode_loop, name="zing-source-decoder", daemon=True
        )
        self._thread.start()

    @property
    def buffered_latents(self):
        return self.accepted_latents - self.submitted_latents

    def check(self):
        if self.failure is not None:
            raise RuntimeError("Decoder generation failed") from self.failure

    def observe_latent(self, latent):
        height, width = int(latent.shape[-2]), int(latent.shape[-1])
        geometry = (height * self.spatial_ratio, width * self.spatial_ratio)
        if self.height is not None and (self.height, self.width) != geometry:
            raise ValueError("Decoder source geometry changed within one session")
        self.height, self.width = geometry

    def append(self, latent, *, latent_start, valid_frame_end, source_dependency_end):
        if self.total_frames is not None:
            raise ValueError("Decoder stream is closed")
        count = check_latent_append(
            latent=latent,
            latent_start=latent_start,
            source_dependency_end=source_dependency_end,
            valid_frame_end=valid_frame_end,
            accepted_latents=self.accepted_latents,
            valid_frame_end_so_far=self.valid_frame_end,
            protocol=self.protocol,
            capacity_latents=self.capacity_latents,
            buffered_latents=self.buffered_latents,
        )
        self.observe_latent(latent)
        # The decode thread runs on its own stream so its tile collectives never
        # queue behind the next latent broadcast; the event orders the handoff.
        ready = None
        if latent.is_cuda:
            ready = torch.cuda.Event()
            ready.record()
        self._pending.append((latent, ready))
        self.accepted_latents = count
        self.valid_frame_end = valid_frame_end

    def submit(self, frames, reference):
        """Move the pending latents into one decode job without waiting for it."""
        pending = self.valid_frame_end - self.submitted_frames
        if frames != pending:
            raise ValueError("Decoder output length does not match its frames")
        job = dict(
            latents=self._pending,
            frames=frames,
            reference=reference,
            done=threading.Event(),
        )
        self._pending = []
        self.submitted_latents = self.accepted_latents
        self.submitted_frames = self.valid_frame_end
        self._put(job)
        self._last_job = job
        return job

    def wait_for_transfer(self):
        """Finish tile collectives before receiving the next latent on another group."""
        if self._last_job is not None:
            if not self._last_job["done"].wait(timeout=self.publisher.timeout):
                raise TimeoutError("Decoder did not finish before the next transfer")
        self.check()

    def finish(self, total_frames):
        """Close the input side; the last dispatch was already accepted whole."""
        if self.total_frames is not None:
            raise ValueError("Decoder stream is already closed")
        if type(total_frames) is not int or total_frames != self.valid_frame_end:
            raise ValueError("Final video frame count does not match latent count")
        if not self.accepted_latents:
            raise ValueError("Decoder cannot finish an empty stream")
        self.total_frames = total_frames

    def _put(self, job):
        try:
            self._jobs.put(job, timeout=self.publisher.timeout)
        except queue.Full as exc:
            raise TimeoutError("Decoder output queue is full") from exc

    def _decode_loop(self):
        index = 0
        stream = None
        if self.device.type == "cuda":
            # The current device and stream are per thread.
            torch.cuda.set_device(self.device)
            stream = torch.cuda.Stream(device=self.device)
        try:
            while True:
                job = self._jobs.get()
                if job is _DRAIN:
                    return
                error = decoded = None
                try:
                    with (
                        torch.no_grad(),
                        torch.cuda.stream(stream) if stream else nullcontext(),
                    ):
                        for latent, ready in job["latents"]:
                            if ready is not None:
                                ready.wait()
                                latent.record_stream(stream)
                            produced = self._stream.push(latent)
                            self._held = (
                                produced
                                if self._held is None
                                else torch.cat((self._held, produced), dim=2)
                            )
                        decoded = self._take(job["frames"])
                        if job["reference"] is not None:
                            # Convert on the decode stream; the D2H waits for it.
                            self.publisher.publish(job["reference"], decoded)
                        if self._tile_parallel and stream is not None:
                            stream.synchronize()
                except BaseException as exc:
                    error = exc
                try:
                    self._agree(index, error)
                except BaseException as exc:
                    self.failure = error or exc
                    if job["reference"] is not None:
                        self.publisher.fail(job["reference"], self.failure)
                    raise
                finally:
                    job["done"].set()
                index += 1
        except BaseException as failure:
            self.failure = failure
            # Pending dispatches can never decode on a dead stream; wake their
            # consumers now instead of leaving them on the shared-memory wait.
            while True:
                try:
                    job = self._jobs.get_nowait()
                except queue.Empty:
                    break
                if job is not _DRAIN:
                    if job["reference"] is not None:
                        self.publisher.fail(job["reference"], failure)
                    job["done"].set()

    def _take(self, count):
        """Emit the declared valid frames; later frames are padding, as locally."""
        if self._held is None or self._held.shape[2] < count:
            raise ValueError("Lynn decoded fewer frames than the source provides")
        values = self._held[:, :, :count]
        self._held = self._held[:, :, :0].clone()
        return values

    def _agree(self, index, error):
        values = [None] * dist.get_world_size(self.decode_group)
        dist.all_gather_object(
            values, (index, error is not None), group=self.decode_group
        )
        if any(row != values[0] for row in values):
            raise EncoderControlError("Decoder job state differs across ranks")
        if error is not None:
            raise EncoderControlError("Decoder job failed") from error

    def drain(self, timeout):
        if not self._thread.is_alive():
            raise RuntimeError("Decoder stream already failed")
        try:
            self._jobs.put(_DRAIN, timeout=timeout)
        except queue.Full as exc:
            raise TimeoutError("Decoder output queue is full") from exc
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("Decoder did not drain its pending output")
        self.check()

    def dispose(self):
        self._stream.close()


def _output_identity(command):
    session, generation, request = (
        command.get("session"),
        command.get("generation_id"),
        command.get("request_id"),
    )
    chunk = command.get("chunk_index")
    if not all(
        isinstance(value, str) and value for value in (session, generation, request)
    ):
        raise ValueError("Decoder output identity is incomplete")
    if type(chunk) is not int or chunk < 0:
        raise ValueError("Decoder output chunk index is invalid")
    return dict(
        session_id=session,
        generation_id=generation,
        request_id=request,
        chunk_index=chunk,
    )


def load_decoder(decoder_config):
    spec = normalize_decoder_spec(
        decoder_config, video_decoder_backend=None, vae_decode_precision="fp32"
    )
    if spec.type != "lynn_streaming":
        raise ValueError("The decoder service supports only lynn_streaming")
    from sglang.multimodal_gen.runtime.models.vaes.lynn_streaming.loader import (
        load_lynn_decoder,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    return load_lynn_decoder(spec, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--decoder-config", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--startup-timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.startup_timeout_seconds <= 0:
        raise ValueError("Decoder timeouts must be positive")
    ServerArgs, set_global_server_args = initialize_runtime_imports()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout_seconds))
    rank, world = dist.get_rank(), dist.get_world_size()
    idle_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=30))
    control_group = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=args.timeout_seconds)
    )
    startup_group = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=args.startup_timeout_seconds)
    )
    decode_group = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=args.timeout_seconds)
    )
    latent_group = tile_device_group = None
    if world > 1:
        latent_group = dist.new_group(
            backend="nccl", timeout=timedelta(seconds=args.timeout_seconds)
        )
        tile_device_group = dist.new_group(
            backend="nccl", timeout=timedelta(seconds=args.timeout_seconds)
        )
    tile_group = (
        _TileGroup(
            rank=rank,
            world_size=world,
            cpu_group=decode_group,
            device_group=tile_device_group,
        )
        if world > 1
        else None
    )
    control = EncoderCommandControl(
        dist,
        rank=rank,
        socket=None,
        idle_group=idle_group,
        operation_group=control_group,
    )
    module = publisher = service = bridge = token = channel = socket = context = None
    device = torch.device("cuda", torch.cuda.current_device())
    try:
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
            module = load_decoder(json.loads(args.decoder_config))
            # Worker-lifetime: a reference outlives its session until the API
            # ACKs it, which happens only after the final dispatch returns.
            publisher = _OutputPublisher(timeout_seconds=args.timeout_seconds)
            if rank == 0:
                context = zmq.Context()
                socket = context.socket(zmq.REP)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.SNDTIMEO, args.timeout_seconds * 1000)
                socket.bind(f"tcp://127.0.0.1:{args.port}")
                control.socket = socket
        except Exception as exc:
            logging.exception("Decoder startup failed on rank %s", rank)
            error = exc
        control.agree("startup", error=error, group=startup_group)
        print(
            json.dumps({"decoder_service_ready": True, "rank": rank, "world": world}),
            flush=True,
        )
        while True:
            command, _payload = control.next()
            if command is None:
                continue
            op = command["op"]
            started = time.time_ns()
            # The full command and session identity must match before any
            # collective, device input or publish touches shared state.
            control.agree(
                "command", (control.sequence, command, token, service is not None)
            )
            reply, error = None, None
            try:
                if op == "ping":
                    reply = dict(
                        ok=True,
                        world_size=control.world_size,
                        busy=service is not None,
                    )
                elif op == "abort":
                    requested = command.get("session")
                    if service is not None and requested == token:
                        try:
                            service.drain(min(args.timeout_seconds, 30))
                        except (TimeoutError, RuntimeError):
                            raise ValueError("Decoder could not drain before abort")
                        service.dispose()
                        service = token = channel = None
                        reply = dict(ok=True, session=requested, released=True)
                    else:
                        reply = dict(
                            ok=True,
                            session=requested,
                            released=False,
                            current_owner_unchanged=True,
                        )
                elif op == "open":
                    if service is not None:
                        reply = dict(
                            ok=False,
                            code="DECODER_BUSY",
                            error="The decoder is busy. Please try again later.",
                        )
                    else:
                        error = None
                        try:
                            requested = command["session"]
                            if not isinstance(requested, str) or not requested:
                                raise ValueError("Invalid decoder session")
                            capacity = command["capacity_latents"]
                            if type(capacity) is not int or capacity < 1:
                                raise ValueError("Invalid decoder latent capacity")
                            service = _DecodeService(
                                module,
                                capacity_latents=capacity,
                                group=tile_group,
                                publisher=publisher,
                                decode_group=decode_group,
                                device=device,
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
                            token = requested
                            channel = control.sequence
                        except Exception as exc:
                            logging.exception(
                                "Decoder open preparation failed on rank %s", rank
                            )
                            error = exc
                        control.agree("open_prepared", (token, channel), error=error)
                        if error is None:
                            reply = dict(ok=True, channel=channel)
                else:
                    # Validate the channel once at the dispatch boundary; handlers
                    # below trust it rather than re-checking session identity.
                    if service is None or command.get("channel") != channel:
                        raise ValueError("Decoder channel does not match")
                    if op not in ("append", "output", "finish", "close"):
                        raise ValueError("Unknown decoder operation")
                    health_error = None
                    try:
                        service.check()
                    except Exception as exc:
                        health_error = exc
                    control.agree("service_checked", error=health_error)
                    if op == "append":
                        shape = command["shape"]
                        if (
                            not isinstance(shape, list)
                            or any(type(n) is not int or n < 1 for n in shape)
                            or shape[0] != 1
                            or shape[1] != 24
                        ):
                            raise ValueError("Invalid decoder latent shape")
                        dtype = getattr(torch, command["dtype"])
                        if world > 1:
                            preparation_error = None
                            try:
                                service.wait_for_transfer()
                            except Exception as exc:
                                preparation_error = exc
                            control.agree("append_prepared", error=preparation_error)
                        if rank == 0:
                            latent = receive_latent(bridge, shape, dtype, device)
                        else:
                            latent = torch.empty(shape, dtype=dtype, device=device)
                        if world > 1:
                            dist.broadcast(latent, src=0, group=latent_group)
                        service.append(
                            latent,
                            latent_start=command["latent_start"],
                            valid_frame_end=command["valid_frame_end"],
                            source_dependency_end=command["source_dependency_end"],
                        )
                        del latent
                    elif op == "output":
                        frames = command["frames"]
                        if type(frames) is not int or frames < 0:
                            raise ValueError("Invalid decoder output length")
                        if frames != service.valid_frame_end - service.submitted_frames:
                            raise ValueError(
                                "Decoder output length does not match its frames"
                            )
                        reference = metadata = None
                        if frames:
                            if rank == 0:
                                reference = publisher.reserve(
                                    frames=frames,
                                    width=service.width,
                                    height=service.height,
                                    identity=_output_identity(command),
                                )
                                metadata = dict(
                                    format="rgb24",
                                    width=service.width,
                                    height=service.height,
                                    channels=3,
                                    bytes_per_frame=(
                                        service.width * service.height * 3
                                    ),
                                )
                            service.submit(frames, reference)
                        reply = dict(ok=True, reference=reference, metadata=metadata)
                    elif op == "finish":
                        service.finish(command["total_frames"])
                    elif op == "close":
                        service.drain(args.timeout_seconds)
                        service.dispose()
                        service = token = channel = None
            except EncoderControlError:
                raise
            except Exception as exc:
                logging.exception("Decoder operation failed on rank %s", rank)
                error = exc
            # A failed execution ends the service: no peer may reuse a possibly
            # damaged stream or communicator, even if its local work succeeded.
            control.agree("executed", control.sequence, error=error)
            # Payload-free ops reply ok only after every rank agreed no error.
            if reply is None:
                reply = dict(ok=True)
            try:
                control.reply(reply)
            except Exception:
                logging.exception("Decoder reply failed on rank %s", rank)
                raise
            control.agree(f"{op}_complete", control.sequence)
            if op not in ("open", "abort", "ping") and reply and reply.get("ok"):
                print(
                    json.dumps(
                        dict(
                            decoder_step=op,
                            session=token,
                            rank=rank,
                            host_start_ns=started,
                            host_end_ns=time.time_ns(),
                            gpu_allocated_bytes=torch.cuda.memory_allocated(),
                            gpu_reserved_bytes=torch.cuda.memory_reserved(),
                            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            decoder_latents=(
                                service.buffered_latents if service else 0
                            ),
                        )
                    ),
                    flush=True,
                )
    except Exception:
        logging.exception("Decoder service stopped on rank %s", rank)
        if rank == 0 and control.request_pending:
            try:
                control.reply(
                    dict(
                        ok=False,
                        code="DECODER_RESTART_REQUIRED",
                        error="The decoder must be restarted before another operation.",
                    )
                )
            except Exception:
                logging.exception("Decoder failure reply could not be sent")
        raise
    finally:
        # Local disposal is safe; do not issue another collective after a
        # communication failure.
        if service is not None:
            service.dispose()
        if publisher is not None:
            publisher.close()
        if socket is not None:
            socket.close()
        if context is not None:
            context.term()


if __name__ == "__main__":
    main()
