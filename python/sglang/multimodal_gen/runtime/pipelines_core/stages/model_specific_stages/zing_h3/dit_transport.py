# SPDX-License-Identifier: Apache-2.0
"""Loopback ZMQ metadata and native tensors over the existing NCCL bridge.

One owner serializes socket operations. Receive the ZMQ descriptor
before posting NCCL receive: an idle NCCL receive can block a model's device-wide
synchronize. Pure control messages never allocate CUDA tensors.
"""

import math
from contextlib import contextmanager
from datetime import timedelta
from threading import Lock

import msgspec.msgpack
import torch
import torch.distributed as dist
import zmq

from sglang.multimodal_gen.runtime.disaggregation.transport.codec import (
    dtype_to_str,
    str_to_dtype,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.nccl_bridge import (
    create_bridge_group,
    receive_bridge_tensor,
    send_bridge_tensor,
    validate_bridge_tensor,
)


class EncoderDiTChannel:
    def __init__(
        self,
        group,
        socket,
        *,
        rank,
        device,
        max_bytes=256 << 20,
        stream_key=None,
    ):
        self.group, self.socket = group, socket
        self.peer, self.device = 1 - rank, device
        self.max_bytes = max_bytes
        self.stream_key = self.peer if stream_key is None else stream_key
        self.sent = self.received = 0
        self.store = self.context = None
        self.aborted = False
        self._group_close_lock = Lock()

    @contextmanager
    def io_context(self):
        """DiT I/O owner only; Encoder's existing execution path is unchanged."""
        if torch.device(self.device).type == "cuda":
            from sglang.multimodal_gen.runtime.utils.zing_capture_stream import (
                dit_io_stream,
            )

            with (
                torch.cuda.device(self.device),
                torch.cuda.stream(
                    dit_io_stream(
                        device=self.device,
                        key=getattr(self, "stream_key", self.peer),
                    )
                ),
            ):
                yield
        else:
            yield

    @staticmethod
    def _cuda_tensors(value):
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from EncoderDiTChannel._cuda_tensors(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from EncoderDiTChannel._cuda_tensors(child)

    def finish_transfer(self, message):
        # Work.wait() orders the calling CUDA stream, not necessarily the CPU.
        # Complete that stream before passing tensor ownership to another thread.
        # Do not synchronize the entire device, or allocate tensors for controls.
        if next(self._cuda_tensors(message), None) is not None:
            torch.cuda.current_stream(self.device).synchronize()

    def prepare_send(self, message):
        # Runs on the model thread before transferring its output to the I/O
        # thread. Only this producer stream must have finished writing it.
        if isinstance(message, dict) and message.pop("_stream_synced", False):
            return
        self.finish_transfer(message)

    def adopt_received(self, message):
        for value in self._cuda_tensors(message):
            value.record_stream(torch.cuda.current_stream(value.device))

    @classmethod
    def connect(
        cls,
        port,
        *,
        rank,
        device="cuda",
        timeout=120,
        backend="nccl",
        host="127.0.0.1",
        bind_host="127.0.0.1",
        store_host="127.0.0.1",
        store=None,
        worker_group_id=None,
        identities=None,
    ):
        # PAIR supports multiple latent replies followed by the input receipt.
        # No routing layer is needed for this single-peer deployment.
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.setsockopt(zmq.LINGER, 1000)
        socket.setsockopt(zmq.SNDHWM, 10)
        socket.setsockopt(zmq.RCVHWM, 10)
        socket.setsockopt(zmq.MAXMSGSIZE, 1 << 20)
        socket.setsockopt(zmq.SNDTIMEO, int(timeout * 1000))
        socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        group = None
        try:
            endpoint = f"tcp://{host if rank == 0 else bind_host}:{port}"
            if rank == 0:
                import secrets

                epoch = secrets.token_hex(16)
                store = store or dist.TCPStore(
                    store_host,
                    0,
                    2,
                    True,
                    timeout=timedelta(seconds=timeout),
                    wait_for_workers=False,
                )
                socket.connect(endpoint)
                socket.send_json(
                    dict(
                        type="bridge.open",
                        version=1,
                        port=store.port,
                        host=store_host,
                        epoch=epoch,
                        worker_group_id=worker_group_id,
                        identities=identities,
                    )
                )
            else:
                socket.bind(endpoint)
                hello = socket.recv_json()
                if (
                    hello.get("type") != "bridge.open"
                    or hello.get("version") != 1
                    or hello.get("worker_group_id") != worker_group_id
                    or hello.get("identities") != identities
                    or type(hello.get("port")) is not int
                    or not 0 < hello["port"] < 65536
                ):
                    raise ValueError("Invalid Encoder/DiT bridge handshake")
                epoch = hello.get("epoch", "legacy")
                store = dist.TCPStore(
                    hello.get("host", "127.0.0.1"),
                    hello["port"],
                    2,
                    False,
                    timeout=timedelta(seconds=timeout),
                )
            group = create_bridge_group(
                store,
                rank,
                prefix="zing-encoder-dit-v1:" + epoch,
                backend=backend,
                timeout_seconds=timeout,
            )
            # Materialize the communicator once at startup using a real tensor.
            # This is the only synthetic transfer; session metadata uses ZMQ.
            if rank == 0:
                reply = socket.recv_json()
                if reply != dict(type="bridge.ready", version=1):
                    raise ValueError("Encoder/DiT link protocol does not match")
                initialization = torch.zeros(1, device=device)
                send_bridge_tensor(group, initialization, peer=1)
            else:
                socket.send_json(dict(type="bridge.ready", version=1))
                initialization = receive_bridge_tensor(
                    group, [1], torch.float32, device, peer=0
                )
            from sglang.multimodal_gen.runtime.realtime.worker_connection_ports import (
                confirm_bridge_connection,
            )

            confirm_bridge_connection(
                store,
                "zing-encoder-dit-v1:" + epoch,
                rank,
                initialization.device,
                timeout=timeout,
            )
            channel = cls(
                group,
                socket,
                rank=rank,
                device=device,
                stream_key=f"{host if rank == 0 else bind_host}:{port}:{rank}",
            )
            channel.store, channel.context = store, context
            return channel
        except BaseException:
            from sglang.multimodal_gen.runtime.realtime.cleanup_watchdog import (
                CleanupWatchdog,
            )

            with CleanupWatchdog(timeout):
                if group is not None:
                    from sglang.multimodal_gen.runtime.disaggregation.transport.nccl_bridge import (
                        abort_bridge_group,
                    )

                    abort_bridge_group(group)
                socket.close()
                context.term()
            raise

    def send(self, message):
        tensors, descriptors = [], []

        def encode(value):
            if isinstance(value, torch.Tensor):
                validate_bridge_tensor(
                    value.shape,
                    value.dtype,
                    expected_ndim=None,
                    expected_batch=None,
                    allowed_dtypes=None,
                    max_bytes=self.max_bytes,
                )
                index = len(tensors)
                tensors.append(value)
                descriptors.append(
                    dict(shape=list(value.shape), dtype=dtype_to_str(value.dtype))
                )
                return {"__tensor__": index}
            if isinstance(value, dict):
                if "__tensor__" in value:
                    raise ValueError("Reserved tensor field in bridge message")
                return {k: encode(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(v) for v in value]
            return value

        body = encode(message)
        header = msgspec.msgpack.encode(
            dict(sequence=self.sent, body=body, tensors=descriptors)
        )
        total = len(header) + sum(t.numel() * t.element_size() for t in tensors)
        if (
            not 0 < len(header) <= 1 << 20
            or len(tensors) > 64
            or total > self.max_bytes
        ):
            raise ValueError("Encoder/DiT message exceeds its capacity")
        self.socket.send(header)
        for value in tensors:
            send_bridge_tensor(self.group, value.to(self.device), peer=self.peer)
        self.sent += 1

    def poll(self):
        """Check only ZMQ metadata readiness; never post an idle GPU receive."""
        return bool(self.socket.poll(timeout=0))

    def receive(self):
        raw = self.socket.recv()
        size = len(raw)
        if not 0 < size <= min(1 << 20, self.max_bytes):
            raise ValueError("Invalid Encoder/DiT message header size")
        header = msgspec.msgpack.decode(raw)
        if header["sequence"] != self.received or len(header["tensors"]) > 64:
            raise ValueError("Invalid Encoder/DiT message sequence or tensor count")
        total, specs = size, []
        for spec in header["tensors"]:
            dtype = str_to_dtype(spec["dtype"])
            validate_bridge_tensor(
                spec["shape"],
                dtype,
                expected_ndim=None,
                expected_batch=None,
                allowed_dtypes=None,
                max_bytes=self.max_bytes,
            )
            total += (
                math.prod(spec["shape"]) * torch.empty((), dtype=dtype).element_size()
            )
            if total > self.max_bytes:
                raise ValueError("Encoder/DiT message exceeds its capacity")
            specs.append((spec["shape"], dtype))
        tensors = [
            receive_bridge_tensor(self.group, shape, dtype, self.device, peer=self.peer)
            for shape, dtype in specs
        ]

        def decode(value):
            if isinstance(value, dict):
                if set(value) == {"__tensor__"}:
                    index = value["__tensor__"]
                    if type(index) is not int or not 0 <= index < len(tensors):
                        raise ValueError("Invalid Encoder/DiT tensor reference")
                    return tensors[index]
                return {k: decode(v) for k, v in value.items()}
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value

        message = decode(header["body"])
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise ValueError("Invalid Encoder/DiT message")
        self.received += 1
        return message

    def abort(self):
        # May be called by the owner thread to unblock an I/O receive. Socket
        # teardown remains exclusively on the I/O thread.
        with self._group_close_lock:
            if self.group is None or self.aborted:
                return
            group = self.group
            # Publish the fault before releasing the lock or entering native
            # code. An abort failure must never fall back to graceful shutdown.
            self.aborted = True
        from sglang.multimodal_gen.runtime.disaggregation.transport.nccl_bridge import (
            abort_bridge_group,
        )

        abort_bridge_group(group)

    def close(self):
        # Private ProcessGroups are not registered in the default world.
        with self._group_close_lock:
            group, self.group = self.group, None
            shutdown = None if self.aborted else getattr(group, "shutdown", None)
        # Native operations may block. Claim teardown under the lock, but do
        # not hold it while calling NCCL. The local reference owns the group
        # until this call returns, even if another thread observes it closed.
        if shutdown is not None:
            shutdown()
        self.store = None
        self.socket.close()
        if self.context is not None:
            self.context.term()
