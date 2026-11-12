# SPDX-License-Identifier: Apache-2.0
"""Dedicated Lynn decoder service on its own GPU group.

The DiT leader sends normalized latents over a private NCCL bridge and receives
one shared-memory reference per dispatch instead of pixels. Decode, uint8
conversion, D2H and publication stay on the decoder host, so the DiT never
waits for decode. One active session per independent listener; model/transport
failures end the service because Lynn K/V state is not retryable. Output
cancellation drains queued inputs before the session owner releases the stream.
"""

import argparse
import asyncio
import json
import logging
import os
import queue
import threading
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import zmq

from sglang.multimodal_gen.configs.pipeline_configs.zing_h3_decoder import (
    normalize_decoder_spec,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.shm_publisher import (
    SharedMemoryFramePublisher,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.worker_control import (
    RoleCommandControl,
    RoleControlError,
)
from sglang.multimodal_gen.runtime.realtime.polling_task_queue import PollingTaskQueue
from sglang.multimodal_gen.runtime.realtime.zing_source_stream import (
    remote_decode_progressive,
)
from sglang.multimodal_gen.runtime.utils.bounded_process_log import diagnostic_print

from .decoder_transport import bridge_group, receive_latent
from .source_decoder import check_latent_append

EncoderCommandControl = RoleCommandControl
EncoderControlError = RoleControlError
_OutputPublisher = SharedMemoryFramePublisher

_DRAIN = object()


def _queue_chunk_timing(entry, command, timeout_seconds):
    """Validate the generation and enqueue timing behind its RGB descriptor.

    Called on rank zero only. This sideband carries no latent/RGB tensors and
    does not submit another decode job.
    """
    direct_output = entry.get("output_session")
    if (
        direct_output is None
        or command.get("generation_id") != direct_output.config["generation_id"]
    ):
        raise ValueError("Decoder chunk timing identity is invalid")
    direct_output.put(
        dict(
            type="decoder.timing",
            request_id=command.get("request_id"),
            chunk_index=command.get("chunk_index"),
            output_frame_start=command.get("output_frame_start"),
            num_frames=command.get("num_frames"),
            frame_stage_breakdown=command.get("frame_stage_breakdown"),
        ),
        timeout_seconds,
        discard_if_stopped=True,
    )


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


class _DecodeStepGate:
    """Serialize multi-session decode steps and decode_group collectives in submit order."""

    def __init__(self):
        self._cond = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0
        self._cancelled: set[int] = set()

    def issue(self) -> int:
        with self._cond:
            ticket = self._next_ticket
            self._next_ticket += 1
            return ticket

    def enter(self, ticket: int) -> None:
        with self._cond:
            while self._serving_ticket != ticket:
                self._cond.wait()

    def leave(self, ticket: int) -> None:
        with self._cond:
            if ticket == self._serving_ticket:
                self._serving_ticket += 1
                while self._serving_ticket in self._cancelled:
                    self._cancelled.remove(self._serving_ticket)
                    self._serving_ticket += 1
                self._cond.notify_all()
            elif ticket > self._serving_ticket:
                self._cancelled.add(ticket)


class _DecodeService:
    """Input boundaries on the command thread, stream state on the decode thread."""

    _DEFAULT_STEP_GATE = _DecodeStepGate()

    def __init__(
        self,
        module,
        *,
        capacity_latents,
        group,
        publisher,
        decode_group,
        device,
        step_gate=None,
        queue_driven=False,
        progressive=False,
    ):
        self.progressive = progressive
        self.queue_driven = queue_driven
        self.protocol = module.protocol
        self.device = device
        self.spatial_ratio = module.spatial_ratio
        self._stream = module.new_stream(frame_limit=None, group=group)
        self.capacity_latents = capacity_latents
        self.publisher = publisher
        self.decode_group = decode_group
        self._step_gate = (
            step_gate if step_gate is not None else self._DEFAULT_STEP_GATE
        )
        self.failure = None
        self.width = self.height = None
        self.accepted_latents = self.submitted_latents = 0
        self.valid_frame_end = self.submitted_frames = 0
        self.total_frames = None
        self._pending = []
        self._held = None
        self._jobs = queue.Queue(maxsize=10)
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

    def output_slices(self, frames):
        start = self.submitted_frames
        counts = []
        for latent_end in range(self.submitted_latents + 1, self.accepted_latents + 1):
            end = min(self.valid_frame_end, self.protocol.frame_count(latent_end))
            if end > start:
                counts.append(end - start)
                start = end
        if sum(counts) != frames:
            raise ValueError("Decoder slices do not match valid output frames")
        return counts

    def submit(self, frames, reference, *, timing_state=None):
        """Move the pending latents into one decode job without waiting for it."""
        pending = self.valid_frame_end - self.submitted_frames
        if frames != pending:
            raise ValueError("Decoder output length does not match its frames")
        if isinstance(timing_state, dict):
            timing_state["submitted_mono_ns"] = time.perf_counter_ns()
        ticket = self._step_gate.issue() if self._step_gate is not None else None
        job = dict(
            latents=self._pending,
            slices=self.output_slices(frames) if self.progressive else None,
            frames=frames,
            reference=reference,
            done=threading.Event(),
            ticket=ticket,
            timing_state=timing_state,
        )
        self._pending = []
        self.submitted_latents = self.accepted_latents
        self.submitted_frames = self.valid_frame_end
        try:
            self._put(job)
        except BaseException:
            if self._step_gate is not None and ticket is not None:
                self._step_gate.leave(ticket)
            raise
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
        if self.queue_driven:
            ticket = self._step_gate.issue() if self._step_gate is not None else None
            job = dict(
                total_frames=total_frames,
                reference=None,
                done=threading.Event(),
                ticket=ticket,
            )
            try:
                self._put(job)
            except BaseException:
                if ticket is not None:
                    self._step_gate.leave(ticket)
                raise
            self._last_job = job

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
            from sglang.multimodal_gen.runtime.utils.zing_capture_stream import (
                decoder_stream,
            )

            torch.cuda.set_device(self.device)
            stream = decoder_stream()

        async def process(tasks):
            nonlocal index
            job = tasks.popleft()
            ticket = job.get("ticket")
            if self._step_gate is not None and ticket is not None:
                self._step_gate.enter(ticket)
            timing_state = job.get("timing_state")
            dec_start_ns = (
                time.perf_counter_ns() if isinstance(timing_state, dict) else 0
            )
            if isinstance(timing_state, dict):
                sub_ns = timing_state.get("submitted_mono_ns", dec_start_ns)
                timing_state["wait_ms"] = max(0.0, (dec_start_ns - sub_ns) / 1e6)
            error = decoded = None
            try:
                with (
                    torch.no_grad(),
                    torch.cuda.stream(stream) if stream else nullcontext(),
                ):
                    if "total_frames" in job:
                        self._stream.finish(job["total_frames"])
                    else:
                        part_index = 0
                        for latent, ready in job["latents"]:
                            if ready is not None:
                                ready.wait()
                                latent.record_stream(stream)
                            pieces = (
                                latent.split(1, dim=2)
                                if job["slices"] is not None
                                else [latent]
                            )
                            for piece in pieces:
                                produced = self._stream.push(piece)
                                self._held = (
                                    produced
                                    if self._held is None
                                    else torch.cat((self._held, produced), dim=2)
                                )
                                if job["slices"] is not None and part_index < len(
                                    job["slices"]
                                ):
                                    decoded = self._take(job["slices"][part_index])
                                    if job["reference"] is not None:
                                        self.publisher.publish_slice(
                                            job["reference"], part_index, decoded
                                        )
                                    part_index += 1
                        if job["slices"] is None:
                            decoded = self._take(job["frames"])
                            if job["reference"] is not None:
                                self.publisher.publish(job["reference"], decoded)
                        elif part_index != len(job["slices"]):
                            raise ValueError(
                                "Decoder did not publish every valid slice"
                            )
                    if self._tile_parallel and stream is not None:
                        stream.synchronize()
                    if isinstance(timing_state, dict):
                        dec_done_ns = time.perf_counter_ns()
                        timing_state["process_ms"] = max(
                            0.0, (dec_done_ns - dec_start_ns) / 1e6
                        )
                        timing_state["done_mono_ns"] = dec_done_ns
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
                if self._step_gate is not None and ticket is not None:
                    self._step_gate.leave(ticket)
                job["done"].set()
            index += 1

        tasks = PollingTaskQueue(process)

        async def consume():
            while True:
                try:
                    job = self._jobs.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(tasks.poll_interval_s)
                    continue
                if job is _DRAIN:
                    return
                tasks.append(job)
                await tasks.run_once()

        try:
            asyncio.run(consume())
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
                    if self._step_gate is not None and job.get("ticket") is not None:
                        self._step_gate.leave(job["ticket"])
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

    @property
    def thread_alive(self) -> bool:
        thread = getattr(self, "_thread", None)
        return bool(thread is not None and thread.is_alive())

    def cancel_pending(self):
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                break
            if job is not _DRAIN:
                if self._step_gate is not None and job.get("ticket") is not None:
                    self._step_gate.leave(job["ticket"])
                job["done"].set()

    def drain(self, timeout):
        if not self._thread.is_alive():
            self.check()
            if (
                self.queue_driven
                and self.total_frames is not None
                and self._stream.closed
            ):
                return
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
        self.cancel_pending()
        if self.thread_alive:
            try:
                self._jobs.put_nowait(_DRAIN)
            except queue.Full:
                pass
            self._thread.join(timeout=0.05)
        if not self.thread_alive:
            self._stream.close()


def _output_identity(command, *, session):
    # Later commands carry a channel; the validated owner was saved by open.
    generation, request = (
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
    parser.add_argument(
        "--port", type=int, default=37000, help="Base ZMQ port; service i uses port + i"
    )
    parser.add_argument("--decoder-config", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--startup-timeout-seconds", type=int, default=600)
    parser.add_argument("--max-sessions", type=int, default=1)
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--advertise-host", default="127.0.0.1")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument(
        "--http-port",
        type=int,
        default=36000,
        help="Base HTTP port; service i uses port + i",
    )
    parser.add_argument("--service-count", type=int, default=1)
    parser.add_argument(
        "--static-connection",
        action="store_true",
        help="Use the integrated launcher's static transport without HTTP group management",
    )
    parser.add_argument(
        "--output-port",
        type=int,
        default=38000,
        help="Base output WS port, separate from management HTTP",
    )
    args = parser.parse_args()

    from sglang.multimodal_gen.runtime.realtime.zing_nccl import configure_zing_nccl

    configure_zing_nccl()

    from sglang.multimodal_gen.runtime.realtime.role_pool.reporting import (
        validate_worker_configuration,
    )

    validate_worker_configuration(
        "decoder", args.service_count, static_connection=args.static_connection
    )
    if args.static_connection:
        args.http_port = args.output_port = None
    import uuid

    from sglang.multimodal_gen.runtime.realtime.worker_connections import (
        ServiceConnections,
        endpoint,
        service_descriptor,
    )
    from sglang.multimodal_gen.runtime.realtime.worker_services import (
        LocalServiceCommands,
        ServiceCommandSockets,
        service_ports,
    )

    service_count = getattr(args, "service_count", 1)
    worker_id = args.worker_id or "decoder-" + uuid.uuid4().hex
    output_port = getattr(args, "output_port", None)
    if output_port is not None and args.http_port is None:
        raise ValueError("output-port requires http-port")
    if service_count > 1 and args.http_port is not None and output_port is None:
        output_port = args.http_port + 1000
    ports = service_ports(
        service_count, command=args.port, http=args.http_port, output=output_port
    )
    max_sessions = getattr(args, "max_sessions", None)
    if max_sessions is None:
        max_sessions = 1
    if (
        args.timeout_seconds <= 0
        or args.startup_timeout_seconds <= 0
        or type(max_sessions) is not int
        or max_sessions < 1
    ):
        raise ValueError("Decoder timeouts and max_sessions must be positive")
    if service_count > 1 and max_sessions != 1:
        raise ValueError("Each independent Decoder service allows one active session")
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
    progressive_output = remote_decode_progressive()
    module = publisher = service = bridge = store = token = channel = socket = (
        context
    ) = None
    # Each listener owns its existing session/bridge maps. Only model weights,
    # rank consensus and the decode execution gate are shared across services.
    lanes = [
        dict(
            sessions={},
            channels={},
            bridges={},
            draining=False,
            max_sessions=max_sessions,
            broker=None,
            publisher=None,
            http=None,
            output_http=None,
        )
        for _ in range(service_count)
    ]
    sessions = lanes[0]["sessions"]
    channels_by_token = lanes[0]["channels"]
    bridges = lanes[0]["bridges"]
    draining = False
    quarantined = False
    output_broker = None
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        error = None
        try:
            if rank == 0 and args.http_port is not None:
                from fastapi import FastAPI

                from sglang.multimodal_gen.runtime.realtime.decoder_output import (
                    DecoderOutputBroker,
                )
                from sglang.multimodal_gen.runtime.realtime.worker_http import (
                    WorkerHTTPServer,
                )
                from sglang.multimodal_gen.runtime.utils.realtime_video import (
                    cancel_async_raw_rgb_frame_reference,
                )

                if max_sessions != 1:
                    raise ValueError(
                        "Direct Decoder output requires one active session"
                    )
                for index, lane in enumerate(lanes):
                    descriptor = service_descriptor(
                        "decoder",
                        worker_id,
                        index,
                        args.advertise_host,
                        http=ports["http"][index],
                        zmq=ports["command"][index],
                        output_ws=ports["output"][index],
                    )

                    def local(op, *, index=index, **fields):
                        reply = socket.submit(
                            index, dict(op=op, **fields), args.timeout_seconds * 3
                        )
                        if not reply.get("ok"):
                            raise RuntimeError("Decoder connection operation failed")
                        return reply

                    manager = ServiceConnections(
                        descriptor,
                        ready=lambda lane=lane: (
                            lane["http"] is not None and lane["http"].ready
                        ),
                        build=lambda group, topology, local=local: local(
                            "group_build", group_id=group, topology=topology
                        ),
                        destroy=lambda local=local: local("group_destroy", force=False),
                        rollback=lambda local=local: local("group_destroy", force=True),
                        cleanup_timeout=args.timeout_seconds,
                    )
                    lane["connections"] = manager
                    broker = DecoderOutputBroker(cancel_async_raw_rgb_frame_reference)
                    lane["broker"] = broker
                    lane["http"] = WorkerHTTPServer(
                        "decoder",
                        args.http_host,
                        ports["http"][index],
                        install=broker.install if output_port is None else None,
                    )
                    # A client disconnect cancels its mailbox; invalid protocol
                    # and real decoder failures retain worker isolation.
                    broker.on_send_failure = manager.failed
                    lane["http"].connections = manager
                    manager.install(lane["http"].app)
                    lane["output_port"] = ports.get("output", ports["http"])[index]
                    if output_port is not None:
                        app = FastAPI()
                        broker.install(app)
                        from sglang.multimodal_gen.runtime.realtime.role_pool.native_http import (
                            install_managed_transport,
                        )

                        install_managed_transport(app, manager)
                        lane["output_http"] = WorkerHTTPServer(
                            "decoder",
                            args.http_host,
                            lane["output_port"],
                            app=app,
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
            module = load_decoder(json.loads(args.decoder_config))
            # Worker-lifetime: a reference outlives its session until the API
            # ACKs it, which happens only after the final dispatch returns.
            for lane in lanes:
                lane["publisher"] = _OutputPublisher(
                    timeout_seconds=args.timeout_seconds, max_in_flight=10
                )
            if rank == 0:
                context = zmq.Context()
                socket = LocalServiceCommands(
                    ServiceCommandSockets(
                        context,
                        ports["command"],
                        timeout_seconds=args.timeout_seconds,
                        router=True,
                        host=args.http_host,
                    )
                )
                socket.guard = lambda index, command: (
                    lanes[index].get("connections") is None
                    or (
                        lanes[index]["connections"].state == "connected"
                        and command.get("worker_group_id")
                        == lanes[index].get("group_id")
                    )
                )
                control.socket = socket
        except Exception as exc:
            logging.exception("Decoder startup failed on rank %s", rank)
            error = exc
        control.agree("startup", error=error, group=startup_group)
        from .codec_warmup import warmup_decoder

        warmup_decoder(
            module,
            group=tile_group,
            agree=lambda stage, value, **kwargs: control.agree(
                stage, value, group=startup_group, **kwargs
            ),
        )
        controllers = [lane["connections"] for lane in lanes if "connections" in lane]
        for lane in lanes:
            if lane["http"] is not None:
                lane["http"].connection_peers = controllers
                lane["connections"].ready = lambda lane=lane: (
                    lane["http"].ready
                    and not any(c.restart_required for c in controllers)
                )
                lane["http"].ready = True
        diagnostic_print(
            json.dumps(
                {
                    "decoder_service_ready": True,
                    "rank": rank,
                    "world": world,
                    "service_count": service_count,
                    "ports": ports,
                }
            ),
            flush=True,
        )
        while True:
            command, _payload = control.next()
            if command is None:
                continue
            service_index = command.pop("_service_index", 0) if service_count > 1 else 0
            lane = lanes[service_index]
            sessions, channels_by_token, bridges = (
                lane["sessions"],
                lane["channels"],
                lane["bridges"],
            )
            draining, max_sessions = lane["draining"], lane["max_sessions"]
            output_broker, publisher = lane["broker"], lane["publisher"]
            op = command["op"]
            started = time.time_ns()
            # The full command and session identity must match before any
            # collective, device input or publish touches shared state.
            control.agree(
                "command",
                (
                    control.sequence,
                    service_index,
                    command,
                    tuple(sorted(channels_by_token.items())),
                    len(sessions),
                    quarantined,
                ),
            )
            reply, error = None, None
            try:
                if quarantined and op not in ("ping", "abort", "bridge_close"):
                    reply = dict(
                        ok=False,
                        code="DECODER_RESTART_REQUIRED",
                        error="The decoder must be restarted before another operation.",
                        restart_required=True,
                    )
                elif op == "group_rejected":
                    reply = dict(
                        ok=False,
                        code="CONNECTION_UNAVAILABLE",
                        error="The worker connection is unavailable.",
                    )
                elif op == "group_build":
                    group_id = command["group_id"]
                    group_error = None
                    try:
                        if rank == 0:
                            host, port = endpoint(
                                command["topology"]["dit"]["endpoints"]["rendezvous"]
                            )
                            store = dist.TCPStore(
                                host,
                                port,
                                None,
                                False,
                                timeout=timedelta(seconds=args.timeout_seconds),
                            )
                            from sglang.multimodal_gen.runtime.realtime.worker_connection_ports import (
                                bridge_epoch,
                            )

                            epoch = bridge_epoch(
                                store,
                                group_id
                                + ":"
                                + command["topology"]["dit"]["service_id"]
                                + ":"
                                + command["topology"]["decoder"]["service_id"],
                                1,
                                timeout=args.timeout_seconds,
                            )
                            bridges[group_id] = bridge_group(
                                store,
                                1,
                                timeout_seconds=args.timeout_seconds,
                                prefix="zing-decoder-group:" + epoch,
                            )
                            from sglang.multimodal_gen.runtime.disaggregation.transport.nccl_bridge import (
                                receive_bridge_tensor,
                            )

                            initialization = receive_bridge_tensor(
                                bridges[group_id], [1], torch.float32, device, peer=0
                            )
                            from sglang.multimodal_gen.runtime.realtime.worker_connection_ports import (
                                confirm_bridge_connection,
                            )

                            confirm_bridge_connection(
                                store,
                                "zing-decoder-group:" + epoch,
                                1,
                                initialization.device,
                                timeout=args.timeout_seconds,
                            )
                            lane["group_store"] = store
                        lane["group_id"] = group_id
                    except Exception as exc:
                        logging.exception("Decoder group construction failed")
                        group_error = exc
                    outcomes = control.gather(group_error is None)
                    reply = dict(ok=all(outcomes))
                elif op == "group_destroy":
                    if command.get("force") and rank == 0:
                        from sglang.multimodal_gen.runtime.disaggregation.transport.nccl_bridge import (
                            abort_bridge_group,
                        )

                        for failed_bridge in bridges.values():
                            abort_bridge_group(failed_bridge)
                    for ch, target in list(sessions.items()):
                        svc = target["service"]
                        direct = target.get("output_session")
                        if not command.get("force"):
                            if svc.accepted_latents and svc.total_frames is None:
                                svc.finish(svc.valid_frame_end)
                            svc.drain(args.timeout_seconds)
                            if (
                                rank == 0
                                and direct is not None
                                and not direct.completed
                                and not direct.stopped.is_set()
                            ):
                                direct.put(
                                    dict(
                                        type="session.completed",
                                        output_frames=svc.valid_frame_end,
                                    ),
                                    args.timeout_seconds,
                                    discard_if_stopped=True,
                                )
                                if not direct.done.wait(args.timeout_seconds) or (
                                    not direct.completed and not direct.stopped.is_set()
                                ):
                                    raise RuntimeError("Decoder output did not finish")
                        if output_broker is not None:
                            output_broker.close(direct)
                        svc.dispose()
                        if svc.thread_alive:
                            raise RuntimeError(
                                "Decoder execution is still active; worker restart is required"
                            )
                        sessions.pop(ch, None)
                        channels_by_token.pop(target["token"], None)
                    if rank == 0:
                        for group in bridges.values():
                            shutdown = getattr(group, "shutdown", None)
                            if shutdown is not None and not command.get("force"):
                                shutdown()
                        bridges.clear()
                        lane["group_store"] = None
                    lane["group_id"] = None
                    service = token = channel = bridge = None
                    reply = dict(ok=True)
                elif op == "ping":
                    reply = dict(
                        ok=True,
                        world_size=control.world_size,
                        busy=quarantined or draining or len(sessions) >= max_sessions,
                        active_sessions=len(sessions),
                        max_sessions=max_sessions,
                        active_bridges=len(bridges),
                        bridge_ids=sorted(bridges.keys()),
                        draining=draining,
                        restart_required=quarantined,
                    )
                elif op == "set_draining":
                    flag = command.get("draining")
                    if type(flag) is not bool:
                        reply = dict(
                            ok=False,
                            code="INVALID_DRAINING_FLAG",
                            error="draining flag must be boolean.",
                            draining=draining,
                            active_sessions=len(sessions),
                        )
                    else:
                        draining = flag
                        reply = dict(
                            ok=True,
                            draining=draining,
                            active_sessions=len(sessions),
                            max_sessions=max_sessions,
                            busy=quarantined
                            or draining
                            or len(sessions) >= max_sessions,
                        )
                elif op == "set_max_sessions":
                    new_max = command.get("max_sessions")
                    if type(new_max) is not int or new_max < 1:
                        reply = dict(
                            ok=False,
                            code="INVALID_MAX_SESSIONS",
                            error="max_sessions must be a positive integer.",
                            max_sessions=max_sessions,
                            active_sessions=len(sessions),
                        )
                    elif new_max != 1 and (
                        service_count > 1 or output_broker is not None
                    ):
                        reply = dict(
                            ok=False,
                            code="INVALID_MAX_SESSIONS",
                            error="An independent Decoder service allows one active session.",
                            max_sessions=max_sessions,
                            active_sessions=len(sessions),
                        )
                    elif new_max < len(sessions):
                        reply = dict(
                            ok=False,
                            code="DECODER_BUSY",
                            error="Cannot reduce max_sessions below active_sessions.",
                            max_sessions=max_sessions,
                            active_sessions=len(sessions),
                        )
                    else:
                        max_sessions = new_max
                        reply = dict(
                            ok=True,
                            max_sessions=max_sessions,
                            active_sessions=len(sessions),
                            busy=quarantined
                            or draining
                            or len(sessions) >= max_sessions,
                        )
                elif op == "bridge_close":
                    bridge_id = command.get("bridge_id")
                    force = bool(command.get("force", False))
                    if not isinstance(bridge_id, str) or not bridge_id:
                        reply = dict(
                            ok=False,
                            code="INVALID_BRIDGE_ID",
                            error="Invalid decoder bridge_id.",
                        )
                    else:
                        in_use_channels = [
                            ch
                            for ch, v in sorted(sessions.items())
                            if v.get("bridge_id") == bridge_id
                        ]
                        if in_use_channels and not force:
                            reply = dict(
                                ok=False,
                                code="DECODER_BUSY",
                                error="Decoder bridge is currently in use by an active session.",
                                bridge_id=bridge_id,
                            )
                        else:
                            drain_timed_out = False
                            for ch in in_use_channels:
                                entry = sessions.get(ch)
                                if entry is None:
                                    continue
                                svc = entry["service"]
                                if output_broker is not None:
                                    output_broker.close(entry.get("output_session"))
                                try:
                                    svc.drain(min(args.timeout_seconds, 30))
                                except TimeoutError:
                                    drain_timed_out = True
                                except RuntimeError:
                                    if getattr(svc, "thread_alive", False):
                                        drain_timed_out = True
                                if drain_timed_out or getattr(
                                    svc, "thread_alive", False
                                ):
                                    drain_timed_out = True
                                    if hasattr(svc, "cancel_pending"):
                                        svc.cancel_pending()
                                    break
                                svc.dispose()
                                sessions.pop(ch, None)
                                channels_by_token.pop(entry["token"], None)
                                if channel == ch:
                                    service = token = channel = bridge = None
                            if drain_timed_out:
                                quarantined = True
                                reply = dict(
                                    ok=False,
                                    code="DECODER_RESTART_REQUIRED",
                                    bridge_id=bridge_id,
                                    closed=False,
                                    released=False,
                                    restart_required=True,
                                    active_bridges=len(bridges),
                                    bridge_ids=sorted(bridges.keys()),
                                    error="Decoder could not drain active session before bridge_close; worker restart is required.",
                                )
                            else:
                                closed = False
                                if rank == 0:
                                    bridge_grp = bridges.pop(bridge_id, None)
                                    if bridge_grp is not None:
                                        closed = True
                                        try:
                                            dist.destroy_process_group(bridge_grp)
                                        except Exception:
                                            pass
                                reply = dict(
                                    ok=True,
                                    bridge_id=bridge_id,
                                    closed=closed,
                                    active_bridges=len(bridges),
                                    bridge_ids=sorted(bridges.keys()),
                                )
                elif op == "bridge_open":
                    error = None
                    bridge_id = None
                    try:
                        bridge_id = command["bridge_id"]
                        if not isinstance(bridge_id, str) or not bridge_id:
                            raise ValueError("Invalid decoder bridge_id")
                        bridge_port = command["bridge_port"]
                        if type(bridge_port) is not int or not 0 < bridge_port < 65536:
                            raise ValueError("Invalid decoder bridge_port")
                        if rank == 0 and bridge_id not in bridges:
                            store = dist.TCPStore(
                                "127.0.0.1",
                                bridge_port,
                                2,
                                False,
                                timeout=timedelta(seconds=args.timeout_seconds),
                            )
                            bridges[bridge_id] = bridge_group(
                                store, 1, timeout_seconds=args.timeout_seconds
                            )
                    except Exception as exc:
                        logging.exception(
                            "Decoder bridge_open preparation failed on rank %s", rank
                        )
                        error = exc
                    control.agree("bridge_open_prepared", bridge_id, error=error)
                    if error is None:
                        reply = dict(ok=True, bridge_id=bridge_id)
                elif op == "abort":
                    requested = command.get("session")
                    target_channel = channels_by_token.get(requested)
                    if target_channel is not None and target_channel in sessions:
                        entry = sessions[target_channel]
                        svc = entry["service"]
                        if output_broker is not None:
                            output_broker.close(entry.get("output_session"))
                        drain_timed_out = False
                        try:
                            svc.drain(min(args.timeout_seconds, 30))
                        except TimeoutError:
                            drain_timed_out = True
                        except RuntimeError:
                            if getattr(svc, "thread_alive", False):
                                drain_timed_out = True
                        if drain_timed_out or getattr(svc, "thread_alive", False):
                            if hasattr(svc, "cancel_pending"):
                                svc.cancel_pending()
                            quarantined = True
                            reply = dict(
                                ok=False,
                                code="DECODER_RESTART_REQUIRED",
                                session=requested,
                                released=False,
                                restart_required=True,
                                error="Decoder could not drain before abort; worker restart is required.",
                            )
                        else:
                            svc.dispose()
                            sessions.pop(target_channel, None)
                            channels_by_token.pop(requested, None)
                            if channel == target_channel:
                                service = token = channel = bridge = store = None
                            entry = svc = None
                            reply = dict(ok=True, session=requested, released=True)
                    else:
                        reply = dict(
                            ok=True,
                            session=requested,
                            released=False,
                            current_owner_unchanged=True,
                        )
                elif op == "open":
                    if (
                        draining
                        or len(sessions) >= max_sessions
                        or command.get("session") in channels_by_token
                    ):
                        reply = dict(
                            ok=False,
                            code="DECODER_BUSY",
                            error="The decoder is busy. Please try again later.",
                        )
                        if draining:
                            reply["draining"] = True
                    else:
                        error = None
                        new_service = new_bridge = new_token = new_channel = None
                        output_session = None
                        try:
                            requested = command["session"]
                            if not isinstance(requested, str) or not requested:
                                raise ValueError("Invalid decoder session")
                            if type(command.get("auto_output", False)) is not bool:
                                raise ValueError("Invalid decoder auto-output mode")
                            if output_broker is not None and command.get(
                                "output_session"
                            ):
                                config = command["output_session"]
                                if config.get(
                                    "session_id"
                                ) != requested or not command.get("auto_output"):
                                    raise ValueError(
                                        "Invalid direct Decoder output session"
                                    )
                                output_session = output_broker.open(config)
                            capacity = command["capacity_latents"]
                            if type(capacity) is not int or capacity < 1:
                                raise ValueError("Invalid decoder latent capacity")
                            new_service = _DecodeService(
                                module,
                                capacity_latents=capacity,
                                group=tile_group,
                                publisher=publisher,
                                decode_group=decode_group,
                                device=device,
                                queue_driven=command.get("auto_output", False),
                                progressive=progressive_output,
                            )
                            if rank == 0:
                                bridge_id = command.get("bridge_id")
                                if bridge_id is not None:
                                    if bridge_id in bridges:
                                        new_bridge = bridges[bridge_id]
                                    elif "bridge_port" in command:
                                        store = dist.TCPStore(
                                            "127.0.0.1",
                                            command["bridge_port"],
                                            2,
                                            False,
                                            timeout=timedelta(
                                                seconds=args.timeout_seconds
                                            ),
                                        )
                                        new_bridge = bridge_group(
                                            store,
                                            1,
                                            timeout_seconds=args.timeout_seconds,
                                        )
                                        bridges[bridge_id] = new_bridge
                                    else:
                                        raise ValueError("Unknown decoder bridge_id")
                                else:
                                    store = dist.TCPStore(
                                        "127.0.0.1",
                                        command["bridge_port"],
                                        2,
                                        False,
                                        timeout=timedelta(seconds=args.timeout_seconds),
                                    )
                                    new_bridge = bridge_group(
                                        store, 1, timeout_seconds=args.timeout_seconds
                                    )
                            new_token = requested
                            new_channel = control.sequence
                        except Exception as exc:
                            if output_broker is not None:
                                output_broker.close(output_session)
                            if new_service is not None:
                                new_service.dispose()
                            logging.exception(
                                "Decoder open preparation failed on rank %s", rank
                            )
                            error = exc
                        control.agree(
                            "open_prepared", (new_token, new_channel), error=error
                        )
                        if error is None:
                            sessions[new_channel] = dict(
                                service=new_service,
                                bridge=new_bridge,
                                bridge_id=command.get("bridge_id"),
                                token=new_token,
                                channel=new_channel,
                                auto_output=command.get("auto_output", False),
                                output_session=output_session,
                            )
                            channels_by_token[new_token] = new_channel
                            service, bridge, token, channel = (
                                new_service,
                                new_bridge,
                                new_token,
                                new_channel,
                            )
                            reply = dict(
                                ok=True,
                                channel=new_channel,
                                source_output_release_protocol="valid-source-frames-v1",
                                progressive_output=service.progressive,
                            )
                            if output_session is not None:
                                reply["oneway_supported"] = True
                                reply["output_attachment"] = dict(
                                    token=output_session.token,
                                    path="/v1/realtime_video/output",
                                    port=lane["output_port"],
                                    url=lane["connections"].service["endpoints"][
                                        "output_ws"
                                    ],
                                )
                            # The session now owns these references. Keeping open
                            # temporaries alive delays anonymous bridge teardown.
                            new_service = new_bridge = None
                else:
                    # Validate the channel once at the dispatch boundary; handlers
                    # below trust it rather than re-checking session identity.
                    target_channel = command.get("channel")
                    entry = sessions.get(target_channel)
                    if entry is None:
                        raise ValueError("Decoder channel does not match")
                    if command.get("oneway") and not (
                        entry.get("auto_output") and entry.get("output_session")
                    ):
                        raise ValueError("One-way input requires direct Decoder output")
                    service = entry["service"]
                    bridge = entry["bridge"]
                    token = entry["token"]
                    channel = target_channel
                    if op not in (
                        "append",
                        "output",
                        "chunk_timing",
                        "source_progress",
                        "finish",
                        "close",
                    ):
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
                        if world > 1 or progressive_output:
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
                    if op == "output" or (op == "append" and entry.get("auto_output")):
                        frames = (
                            service.valid_frame_end - service.submitted_frames
                            if op == "append"
                            else command["frames"]
                        )
                        if type(frames) is not int or frames < 0:
                            raise ValueError("Invalid decoder output length")
                        if frames != service.valid_frame_end - service.submitted_frames:
                            raise ValueError(
                                "Decoder output length does not match its frames"
                            )
                        reference = metadata = None
                        direct_output = entry.get("output_session")
                        if frames:
                            if rank == 0:
                                if direct_output is not None and (
                                    command.get("generation_id")
                                    != direct_output.config["generation_id"]
                                    or not isinstance(
                                        command.get("output_metadata"), dict
                                    )
                                ):
                                    raise ValueError(
                                        "Decoder output metadata or generation is invalid"
                                    )
                                reference = publisher.reserve(
                                    frames=frames,
                                    width=service.width,
                                    height=service.height,
                                    identity=_output_identity(command, session=token),
                                    **(
                                        {"frame_slices": service.output_slices(frames)}
                                        if getattr(service, "progressive", False)
                                        else {}
                                    ),
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
                            submitted = False
                            timing_state = (
                                {}
                                if direct_output is not None
                                and direct_output.config.get(
                                    "return_stage_timing", False
                                )
                                else None
                            )
                            try:
                                service.submit(
                                    frames, reference, timing_state=timing_state
                                )
                                submitted = True
                                if direct_output is not None:
                                    direct_output.put(
                                        dict(
                                            type="decoder.frames",
                                            reference=reference,
                                            metadata=metadata,
                                            output_metadata=command["output_metadata"],
                                            timing_state=timing_state,
                                        ),
                                        args.timeout_seconds,
                                        discard_if_stopped=True,
                                    )
                            except BaseException as exc:
                                if reference is not None:
                                    if direct_output is not None and submitted:
                                        # Submit may already own this reference. Cancel
                                        # consumption rather than publishing it twice.
                                        output_broker.cancel_reference(reference)
                                    else:
                                        publisher.fail(reference, exc)
                                        if direct_output is not None:
                                            output_broker.cancel_reference(reference)
                                raise
                        reply = dict(
                            ok=True,
                            reference=None if direct_output is not None else reference,
                            metadata=metadata,
                        )
                        if direct_output is not None:
                            reply["output_queued"] = True
                    elif op == "chunk_timing":
                        if rank == 0:
                            _queue_chunk_timing(entry, command, args.timeout_seconds)
                    elif op == "source_progress":
                        if rank == 0:
                            direct_output = entry.get("output_session")
                            if (
                                direct_output is None
                                or command.get("generation_id")
                                != direct_output.config["generation_id"]
                            ):
                                raise ValueError(
                                    "Decoder source progress identity is invalid"
                                )
                            direct_output.source_progress(
                                command["owned_frame_end"],
                                command["source_records"],
                                args.timeout_seconds,
                            )
                    elif op == "finish":
                        service.finish(command["total_frames"])
                        if entry.get("auto_output"):
                            service.drain(args.timeout_seconds)
                        direct_output = entry.get("output_session")
                        if (
                            direct_output is not None
                            and not direct_output.stopped.is_set()
                        ):
                            if direct_output.consumed_frames != command["total_frames"]:
                                raise ValueError(
                                    "Decoder source credit did not drain before EOS"
                                )
                            direct_output.put(
                                dict(
                                    type="session.completed",
                                    output_frames=command["total_frames"],
                                ),
                                args.timeout_seconds,
                                discard_if_stopped=True,
                            )
                            if not direct_output.done.wait(args.timeout_seconds) or (
                                not direct_output.completed
                                and not direct_output.stopped.is_set()
                            ):
                                raise RuntimeError("Decoder output did not complete")
                    elif op == "close":
                        if output_broker is not None:
                            output_broker.close(entry.get("output_session"))
                        service.drain(args.timeout_seconds)
                        service.dispose()
                        del sessions[target_channel]
                        channels_by_token.pop(token, None)
                        entry = None
                        service = token = channel = bridge = store = None
            except EncoderControlError:
                raise
            except Exception as exc:
                logging.exception("Decoder operation failed on rank %s", rank)
                error = exc
            # A failed execution ends the service: no peer may reuse a possibly
            # damaged stream or communicator, even if its local work succeeded.
            if error is not None:
                for worker_lane in lanes:
                    if worker_lane.get("connections") is not None:
                        worker_lane["connections"].quarantine()
                    if worker_lane["http"] is not None:
                        worker_lane["http"].ready = False
                from sglang.multimodal_gen.runtime.realtime.cleanup_watchdog import (
                    CleanupWatchdog,
                )

                # Even the CPU failure agreement can stall behind a peer.
                with CleanupWatchdog(args.timeout_seconds):
                    control.agree("executed", control.sequence, error=error)
            else:
                control.agree("executed", control.sequence)
            # Payload-free ops reply ok only after every rank agreed no error.
            if reply is None:
                reply = dict(ok=True)
            try:
                control.reply(reply)
            except Exception:
                logging.exception("Decoder reply failed on rank %s", rank)
                raise
            control.agree(f"{op}_complete", control.sequence)
            lane["draining"], lane["max_sessions"] = draining, max_sessions
            for worker_lane in lanes:
                if worker_lane["http"] is not None:
                    worker_lane["http"].ready = not quarantined
            if (
                op
                not in (
                    "open",
                    "bridge_open",
                    "bridge_close",
                    "set_max_sessions",
                    "set_draining",
                    "abort",
                    "ping",
                )
                and reply
                and reply.get("ok")
            ):
                diagnostic_print(
                    json.dumps(
                        dict(
                            decoder_step=op,
                            service_index=service_index,
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
        quarantined = True
        for lane in lanes:
            if lane.get("connections") is not None:
                lane["connections"].quarantine()
            if lane["http"] is not None:
                lane["http"].ready = False
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
        from sglang.multimodal_gen.runtime.realtime.cleanup_watchdog import (
            CleanupWatchdog,
        )

        with CleanupWatchdog(args.timeout_seconds):
            if quarantined and rank == 0:
                from sglang.multimodal_gen.runtime.disaggregation.transport.nccl_bridge import (
                    abort_bridge_group,
                )

                for failed_lane in lanes:
                    for failed_bridge in failed_lane["bridges"].values():
                        abort_bridge_group(failed_bridge)
            for lane in lanes:
                if lane["http"] is not None:
                    lane["http"].ready = False
                if lane["broker"] is not None:
                    lane["broker"].close(lane["broker"].session)
            # No cross-rank collective after failure; retain the existing per-stream
            # drain/disposal policy and release every service's output resources.
            for lane in lanes:
                for entry in list(lane["sessions"].values()):
                    if entry.get("service") is not None:
                        entry["service"].dispose()
                lane["sessions"].clear()
                lane["channels"].clear()
                lane["bridges"].clear()
                lane["group_store"] = None
                if lane["publisher"] is not None:
                    lane["publisher"].close()
                if lane["output_http"] is not None:
                    lane["output_http"].close()
                if lane["http"] is not None:
                    lane["http"].close()
            entry = service = bridge = store = None
            new_service = new_bridge = None
            if socket is not None:
                socket.close()
            if context is not None:
                context.term()


if __name__ == "__main__":
    from sglang.multimodal_gen.runtime.realtime.worker_process import run_worker

    run_worker(main)
