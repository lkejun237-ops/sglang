# SPDX-License-Identifier: Apache-2.0
"""Bounded producer for a dedicated, same-node progressive encoder group.

Only the DiT leader owns the RPC/bridge thread. The serving thread replicates
completed moments to DiT ranks before the unchanged posterior sampler runs.
The producer never calls a DiT collective or changes its model/backend flags.
"""

import hashlib
import json
import math
import queue
import sys
import threading
import time
import uuid
from datetime import timedelta

import torch
import torch.distributed as dist
import zmq

from .encoder_recovery import request_abort
from .encoder_transport import bridge_group, receive_moments
from .media_geometry import SourceGeometry
from .progressive_encoder import PROGRESSIVE_ENCODER_VERSION, EncoderStateUsage
from .streaming_codecs import CodecSlice


def available_latents(frames, ended):
    clips, tail = divmod(frames, 17)
    if ended:
        return clips * 5 + (1 + (tail - 1 + 3) // 4 if tail else 0)
    return clips * 5 + (1 if tail else 0) + (2 if tail >= 9 else 0)


class RemoteEncoderSession:
    contract_version = PROGRESSIVE_ENCODER_VERSION
    minimum_credit_frames = 8

    def __init__(self, codecs, *, capacity_frames, config):
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_world_group,
        )

        self.codecs, self.config = codecs, dict(config)
        geometry = config.get("source_geometry")
        self.source_geometry = None if geometry is None else SourceGeometry(**geometry)
        self.resize_host_ms = 0.0
        self.group = get_world_group()
        self.leader = self.group.rank_in_group == 0
        self.device = next(codecs.modules["video_vae"].encoder.parameters()).device
        self.capacity_frames = capacity_frames
        self.capacity_bytes = config.get("capacity_bytes", 256 << 20)
        self.timeout = config.get("timeout_seconds", 120)
        self.admission_endpoint = config.get("admission_endpoint")
        if type(config.get("cross_dispatch_prefetch", False)) is not bool:
            raise ValueError("Cross-dispatch source prefetch flag must be boolean")
        if self.admission_endpoint is not None and not config.get(
            "cross_dispatch_prefetch", False
        ):
            raise ValueError("Source admission requires an enabled prefetch deployment")
        if config.get("cross_dispatch_prefetch", False) and (
            config.get("mode") != "overlap"
            or not isinstance(self.admission_endpoint, str)
            or not self.admission_endpoint.startswith("ipc:///")
        ):
            raise ValueError(
                "Cross-dispatch prefetch requires a private local admission endpoint"
            )
        self.admission_lock = threading.Lock()
        self.admission_digests = {}
        self.prefetched_frames = 0
        self.port = config["port"]
        self.mode = config["mode"]
        if self.mode not in ("serial", "overlap"):
            raise ValueError("Encoder execution mode must be serial or overlap")
        if type(self.port) is not int or not 1024 <= self.port <= 65535:
            raise ValueError("Invalid encoder service port")
        if not 1 <= self.capacity_bytes <= 256 << 20 or not 1 <= self.timeout <= 600:
            raise ValueError("Invalid encoder queue capacity or timeout")
        self.profile_dir = config.get("profile_dir")
        self.stage_timing_dir = config.get("stage_timing_dir")
        self.token = config.get("session_id") or uuid.uuid4().hex
        if not isinstance(self.token, str) or not self.token:
            raise ValueError(
                "Remote encoder session identity must be a nonempty string"
            )
        self.accepted_frames = self.emitted_latents = self._consumed = 0
        self.ended = self.disposed = False
        self.signature = None
        self.commands = queue.Queue(maxsize=capacity_frames + 1)
        # A clip boundary can leave an 8-frame slice followed by the new
        # clip's first frame. Admit the next 8 frames before either is popped.
        result_slots = 3 if self.mode == "overlap" else 2
        self.results = queue.Queue(maxsize=result_slots)
        self.permits = threading.Semaphore(
            result_slots if self.mode == "overlap" else 0
        )
        self.stop = threading.Event()
        self.failure = None
        self.recovery = None
        self.remote_usage = {"retained_bytes": 0, "temporal_positions": 0}
        self.thread = None
        if self.leader:
            self.thread = threading.Thread(
                target=self._produce, name="zing-source-encoder", daemon=True
            )
            self.thread.start()

    def _failure_message(self):
        recovery = getattr(self, "recovery", None) or {}
        if recovery.get("code") == "ENCODER_RESTART_REQUIRED":
            return "Remote encoder failed; worker restart is required before reuse"
        if recovery.get("code") == "ENCODER_RECOVERY_UNCONFIRMED":
            return (
                "Remote encoder failed; recovery is unconfirmed; "
                "check the worker owner and restart state before reuse"
            )
        return "Remote encoder failed"

    def _open(self):
        if self.disposed:
            raise ValueError("Remote encoder is disposed")
        if self.failure is not None:
            raise RuntimeError(self._failure_message()) from self.failure

    @property
    def buffered_frames(self):
        return self.accepted_frames - self._consumed

    @property
    def append_capacity_frames(self):
        """Formal input headroom, independent of background producer progress."""
        if self.ended or self.disposed:
            return 0
        if self.signature is None:
            raise ValueError("Source dimensions are not established")
        capacity = min(
            self.capacity_frames, self.capacity_bytes // (3 * math.prod(self.signature))
        )
        return max(0, capacity - self.buffered_frames)

    @property
    def ready(self):
        return not self.disposed and self.emitted_latents < available_latents(
            self.accepted_frames, self.ended
        )

    def append(self, pixels, *, frame_start):
        self._open()
        if (
            self.ended
            or type(frame_start) is not int
            or frame_start != self.accepted_frames
        ):
            raise ValueError("Source frames must be consecutive and unfinished")
        if (
            pixels.ndim != 4
            or pixels.shape[0] != 3
            or min(pixels.shape) < 1
            or pixels.dtype != torch.uint8
        ):
            raise ValueError("Source pixels must be nonempty RGB bytes [3,T,H,W]")
        signature = tuple(pixels.shape[2:])
        if (
            self.source_geometry is not None
            and signature != self.source_geometry.input_shape
        ):
            raise ValueError("Remote source differs from negotiated input dimensions")
        count = pixels.shape[1]
        if self.signature is not None and signature != self.signature:
            raise ValueError("Source pixel dimensions cannot change")
        retained = (self.buffered_frames + count) * 3 * math.prod(signature)
        if (
            self.buffered_frames + count > self.capacity_frames
            or retained > self.capacity_bytes
        ):
            raise ValueError("Remote encoder input capacity exceeded")
        if self.leader and self.admission_endpoint:
            self._check_admission(
                frame_start,
                pixels.permute(1, 2, 3, 0).contiguous().numpy().tobytes(),
                "dispatch",
            )
        elif self.leader:
            # One bounded copy owns the input until the encoder accepts it.
            self.commands.put(
                ("append", pixels.cpu().contiguous().clone(), frame_start),
                timeout=self.timeout,
            )
        self.signature = signature
        self.accepted_frames += count

    def finish(self):
        self._open()
        if self.ended or self.accepted_frames == 0:
            raise ValueError("Remote encoder requires a nonempty unfinished stream")
        if self.leader and not self.admission_endpoint:
            self.commands.put(("finish", None, None), timeout=self.timeout)
        self.ended = True

    def _check_admission(self, start, data, owner):
        digest = hashlib.sha256(data).hexdigest()
        with self.admission_lock:
            previous = self.admission_digests.pop(start, None)
            if previous is not None:
                if previous[0] == owner or previous[1] != digest:
                    raise ValueError(
                        "Source admission identity does not match the dispatch"
                    )
            else:
                self.admission_digests[start] = (owner, digest)

    def _produce(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout * 1000)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout * 1000)
        socket.connect(f"tcp://127.0.0.1:{self.port}")
        admission = None
        if self.admission_endpoint:
            admission = context.socket(zmq.PULL)
            admission.setsockopt(zmq.LINGER, 0)
            admission.setsockopt(zmq.RCVHWM, self.capacity_frames + 1)
            admission.connect(self.admission_endpoint)
        opened = open_attempted = False
        data_operation_active = False
        rpc_in_flight = False

        def rpc(op, payload=None, **fields):
            nonlocal rpc_in_flight
            request = dict(
                op=op, session=self.token, profile_dir=self.profile_dir, **fields
            )
            parts = [json.dumps(request).encode()]
            if payload is not None:
                parts.append(payload)
            rpc_in_flight = True
            socket.send_multipart(parts)
            reply = socket.recv_json()
            rpc_in_flight = False
            if not reply.get("ok"):
                raise RuntimeError("Encoder service rejected the operation")
            self.remote_usage = reply.get("state", self.remote_usage)
            return reply

        try:
            torch.cuda.set_device(self.device)
            stream = torch.cuda.Stream(device=self.device)
            store = dist.TCPStore(
                "127.0.0.1",
                0,
                2,
                True,
                timeout=timedelta(seconds=self.timeout),
                wait_for_workers=False,
            )
            open_attempted = True
            source_encoding = (
                None if self.source_geometry is None else self.source_geometry.receipt()
            )
            opened_reply = rpc(
                "open",
                bridge_port=store.port,
                capacity_frames=self.capacity_frames,
                stage_timing_dir=self.stage_timing_dir,
                source_encoding=source_encoding,
            )
            opened = True
            data_operation_active = True
            bridge = bridge_group(store, 0, timeout_seconds=self.timeout)
            data_operation_active = False
            if (
                self.source_geometry is not None
                and opened_reply.get("source_encoding") != source_encoding
            ):
                raise ValueError("Encoder did not acknowledge source geometry")
            with torch.cuda.stream(stream):
                while not self.stop.is_set():
                    try:
                        if admission is None:
                            op, pixels, start = self.commands.get(timeout=0.1)
                        else:
                            if not admission.poll(100):
                                continue
                            parts = admission.recv_multipart()
                            header = json.loads(parts[0])
                            op = header["op"]
                            pixels = start = None
                            if op == "append":
                                start, count = (
                                    header["source_frame_start"],
                                    header["num_frames"],
                                )
                                width, height = header["width"], header["height"]
                                if (
                                    self.source_geometry is not None
                                    and (height, width)
                                    != self.source_geometry.input_shape
                                ):
                                    raise ValueError(
                                        "Source admission dimensions changed"
                                    )
                                if (
                                    start != self.prefetched_frames
                                    or not 1 <= count <= 17
                                    or len(parts) != 2
                                    or len(parts[1]) != count * width * height * 3
                                    or len(parts[1]) > self.capacity_bytes
                                    or (self.prefetched_frames - self._consumed + count)
                                    * width
                                    * height
                                    * 3
                                    > self.capacity_bytes
                                    or self.prefetched_frames - self._consumed + count
                                    > self.capacity_frames
                                ):
                                    raise ValueError(
                                        "Invalid or over-capacity source admission"
                                    )
                                self._check_admission(start, parts[1], "producer")
                                pixels = (
                                    torch.frombuffer(
                                        bytearray(parts[1]), dtype=torch.uint8
                                    )
                                    .reshape(count, height, width, 3)
                                    .permute(3, 0, 1, 2)
                                    .contiguous()
                                )
                                self.prefetched_frames += count
                            elif (
                                op != "finish"
                                or header["source_frame_end"] != self.prefetched_frames
                            ):
                                raise ValueError("Invalid source admission end")
                    except queue.Empty:
                        continue
                    if op == "append":
                        if self.source_geometry is not None:
                            started = time.perf_counter()
                            pixels = self.source_geometry.resize(pixels)
                            self.resize_host_ms += (
                                time.perf_counter() - started
                            ) * 1000
                        reply = rpc(
                            op,
                            pixels.numpy().tobytes(),
                            shape=list(pixels.shape),
                            frame_start=start,
                            capacity_bytes=self.capacity_bytes,
                            dispatch_accepted_frames=self.accepted_frames,
                            cross_dispatch_prefetch=admission is not None,
                        )
                    else:
                        reply = rpc(op)
                    del pixels
                    while reply.get("ready") and not self.stop.is_set():
                        if not self.permits.acquire(timeout=0.1):
                            continue
                        if self.stop.is_set():
                            break
                        data_operation_active = True
                        reply = rpc(
                            "pop",
                            dispatch_accepted_frames=self.accepted_frames,
                            prefetched_frames=self.prefetched_frames,
                        )
                        with torch.profiler.record_function("Zing.encoder_receive"):
                            value = receive_moments(
                                bridge,
                                reply["shape"],
                                getattr(torch, reply["dtype"]),
                                self.device,
                            )
                            # Publish only completed storage; no device-wide synchronization.
                            stream.synchronize()
                        data_operation_active = False
                        self.results.put((reply, value), timeout=self.timeout)
        except BaseException as error:
            self.failure = error
        finally:
            if opened and self.failure is None and not rpc_in_flight:
                try:
                    rpc("close")
                except BaseException as error:
                    if self.failure is None:
                        self.failure = error
            if admission is not None:
                admission.close()
            socket.close()
            context.term()
            if open_attempted and self.failure is not None:
                self.recovery = request_abort(
                    self.port,
                    self.token,
                    self.timeout,
                    require_restart=data_operation_active,
                )
                if not (
                    self.recovery.get("released")
                    or self.recovery.get("already_released")
                ):
                    if hasattr(self.failure, "add_note"):
                        self.failure.add_note(
                            "Encoder recovery is unconfirmed; do not reuse the failed "
                            "worker without checking its owner and restart state."
                        )

    def pop(self):
        if self.disposed:
            raise ValueError("Remote encoder is disposed")
        if not self.ready:
            return None
        reply = value = None
        if self.leader:
            if self.mode == "serial":
                self.permits.release()
            try:
                # A failed producer must wake every DiT rank with an error.
                import time

                deadline = time.monotonic() + self.timeout
                while True:
                    self._open()
                    try:
                        reply, value = self.results.get(timeout=0.1)
                        break
                    except queue.Empty:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Encoder result timed out")
            except BaseException:
                reply = {"error": self._failure_message()}
        metadata = [reply]
        dist.broadcast_object_list(
            metadata, src=self.group.ranks[0], group=self.group.cpu_group
        )
        reply = metadata[0]
        if "error" in reply:
            raise RuntimeError(reply["error"])
        if not self.leader:
            value = torch.empty(
                reply["shape"], dtype=getattr(torch, reply["dtype"]), device=self.device
            )
        with torch.profiler.record_function("Zing.encoder_to_dit_ranks"):
            dist.broadcast(
                value, src=self.group.ranks[0], group=self.group.device_group
            )
        if (
            reply["start"] != self.emitted_latents
            or not self._consumed < reply["dependency"] <= self.accepted_frames
        ):
            raise ValueError("Remote encoder returned an invalid slice")
        self.emitted_latents, self._consumed = reply["end"], reply["dependency"]
        if self.leader and self.mode == "overlap":
            self.permits.release()
        return CodecSlice(value, reply["start"], reply["end"], reply["dependency"])

    def state_usage(self):
        if self.disposed:
            return EncoderStateUsage(0, 0)
        retained_frames = max(
            self.buffered_frames, self.prefetched_frames - self._consumed
        )
        raw = retained_frames * 3 * math.prod(self.signature or (0, 0))
        # Remote layer caches plus the two completed moments slots. The raw
        # budget conservatively includes inputs already transferred remotely.
        with self.results.mutex:
            queued = sum(v.numel() * v.element_size() for _, v in self.results.queue)
        return EncoderStateUsage(
            raw + queued + self.remote_usage["retained_bytes"],
            self.remote_usage["temporal_positions"],
        )

    def queue_snapshot(self):
        prefetched_retained = (
            0 if self.disposed else max(0, self.prefetched_frames - self._consumed)
        )
        with self.results.mutex:
            pending = len(self.results.queue)
            result_bytes = sum(
                v.numel() * v.element_size() for _, v in self.results.queue
            )
        return dict(
            source_resize_host_ms_total=self.resize_host_ms,
            prefetch_retained_input_frames=prefetched_retained,
            prefetch_retained_input_bytes=prefetched_retained
            * 3
            * math.prod(self.signature or (0, 0)),
            prefetched_frames=self.prefetched_frames,
            dispatch_accepted_frames=self.accepted_frames,
            admission_pending_identities=len(self.admission_digests),
            pending_results=pending,
            result_bytes=result_bytes,
            pending_commands=self.commands.qsize(),
            retained_input_frames=self.buffered_frames,
            retained_input_bytes=self.buffered_frames
            * 3
            * math.prod(self.signature or (0, 0)),
        )

    def reset(self):
        self._open()
        if self.admission_endpoint:
            self.dispose()
            raise ValueError(
                "Source prefetch reset requires a new session admission generation"
            )
        self.dispose()
        # dispose preserves an active caller exception, but reset must never
        # erase a failed close even when called from an exception handler.
        if self.failure is not None:
            raise RuntimeError("Remote encoder cleanup failed") from self.failure
        self.__init__(
            self.codecs, capacity_frames=self.capacity_frames, config=self.config
        )

    def dispose(self):
        if self.disposed:
            return
        self.stop.set()
        self.permits.release()
        if self.thread is not None:
            # Existing RPC send/recv each have a timeout: allow one outstanding
            # RPC (2), bridge setup/receive (1), normal close RPC (2), and the
            # fresh abort's shared send+recv deadline (1). CUDA/driver hangs can
            # exceed these bounds; the alive-thread check must still fail closed.
            self.thread.join(timeout=6 * self.timeout + 2)
            if self.thread.is_alive():
                raise RuntimeError("Encoder producer did not terminate")
        for q in (self.commands, self.results):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        self.admission_digests.clear()
        self._consumed = self.accepted_frames
        self.remote_usage = {"retained_bytes": 0, "temporal_positions": 0}
        self.disposed = True
        if self.failure is not None:
            primary_error = sys.exc_info()[1]
            if primary_error is None:
                raise RuntimeError("Remote encoder cleanup failed") from self.failure
            # A finally cleanup must not replace the original generation error.
            # Keep failure on the disposed session; no reset can reuse it.
            if hasattr(primary_error, "add_note"):
                primary_error.add_note("Remote encoder cleanup also failed")


def create_session(codecs, *, capacity_frames, config):
    return RemoteEncoderSession(codecs, capacity_frames=capacity_frames, config=config)
