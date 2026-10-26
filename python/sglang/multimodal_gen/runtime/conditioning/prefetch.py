# SPDX-License-Identifier: Apache-2.0
"""Session-private early conditioning RPC; GPU collectives stay on the caller."""

import asyncio
import logging
import threading

import msgspec.msgpack
import zmq
import zmq.asyncio

from .contract import ConditionError

logger = logging.getLogger(__name__)


class PromptPrefetchPublisher:
    """One sending packet plus the latest pending packet, never an event queue."""

    def __init__(self, endpoint, *, max_bytes, reserve):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        try:
            self.socket.bind(endpoint)
        except BaseException:
            self.socket.close()
            self.context.term()
            raise
        self.max_bytes, self.reserve = max_bytes, reserve
        self.pending = self.task = None
        self.closed = False

    def submit(self, command):
        if self.closed:
            return
        packet = msgspec.msgpack.encode(command)
        if len(packet) > self.max_bytes:
            # This is speculative work; the normal worker path remains valid.
            logger.warning("Conditioning prefetch packet exceeds the transport budget")
            return
        try:
            credit = self.reserve(len(packet))
        except Exception:
            logger.exception("Conditioning prefetch admission failed")
            return
        if self.pending is not None:
            self.pending[1].release()
        self.pending = packet, credit
        if self.task is None:
            self.task = asyncio.create_task(self._send())

    async def _send(self):
        try:
            while self.pending is not None and not self.closed:
                packet, credit = self.pending
                self.pending = None
                try:
                    await self.socket.send(packet)
                finally:
                    credit.release()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Conditioning prefetch publication failed")
        finally:
            self.task = None

    def close(self):
        self.closed = True
        if self.pending is not None:
            self.pending[1].release()
            self.pending = None
        if self.task is not None:
            self.task.cancel()
        self.socket.close()
        self.context.term()


class ConditioningPrefetch:
    """One CPU receiver/encoder thread and one digest-checked CPU result.

    The same lock serializes boundary fallback and speculative RPC. No SP
    collective, CUDA allocation, or active prompt mutation runs in this thread.
    """

    def __init__(self, endpoint, *, prepare, identity, client, max_bytes):
        if not isinstance(endpoint, str) or not endpoint.startswith("ipc:///"):
            raise ValueError("Conditioning prefetch requires a private local endpoint")
        self.prepare, self.identity, self.client = prepare, identity, client
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.cached = None
        self.hits = self.encodes = self.failures = self.fallbacks = 0
        self.thread = threading.Thread(
            target=self._receive,
            args=(endpoint,),
            name="zing-qwen-prefetch",
            daemon=True,
        )
        self.thread.start()

    def encode(self, prompt, media, *, input_digest):
        # If the receiver is already encoding this input, wait only for its
        # remaining work, then reuse the CPU result. Verify against the actual
        # worker-normalized input, never merely an event ID.
        deadline = self.client.clock() + self.client.timeout_s
        if not self.lock.acquire(timeout=self.client.timeout_s):
            raise ConditionError("CONDITION_TIMEOUT")
        try:
            if self.stop.is_set():
                raise RuntimeError("Conditioning prefetch is closed")
            if self.client.clock() >= deadline:
                raise ConditionError("CONDITION_TIMEOUT")
            if self.cached is not None and self.cached[0] == input_digest:
                self.hits += 1
                return self.cached[1]
            self.fallbacks += 1
            value = self.client.encode(
                prompt, media, input_digest=input_digest, deadline=deadline
            )
            self.cached = input_digest, value
            return value
        finally:
            self.lock.release()

    def _receive(self, endpoint):
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.MAXMSGSIZE, self.max_bytes)
        socket.connect(endpoint)
        try:
            while not self.stop.is_set():
                if not socket.poll(100):
                    continue
                packet = socket.recv()
                if len(packet) > self.max_bytes:
                    continue
                try:
                    command = msgspec.msgpack.decode(packet)
                    if not isinstance(command, dict) or command.get("rejected"):
                        continue
                    with self.lock:
                        if self.stop.is_set():
                            break
                        prompt, media = self.prepare(command)
                        digest = self.identity(prompt, media)
                        if self.stop.is_set():
                            break
                        # A boundary fallback may have finished before this
                        # packet was received. Do not encode the same input again.
                        if self.cached is not None and self.cached[0] == digest:
                            continue
                        result = self.client.encode(prompt, media, input_digest=digest)
                        self.encodes += 1
                        if not self.stop.is_set():
                            self.cached = digest, result
                except Exception:
                    self.failures += 1
                    # The authoritative boundary path repeats validation and
                    # returns the existing public error without touching active KV.
                    logger.exception(
                        "Early conditioning failed; using boundary validation"
                    )
        finally:
            socket.close()
            context.term()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=self.client.timeout_s + 2)
        if self.thread.is_alive():
            raise RuntimeError(
                "Conditioning prefetch did not drain before session cleanup"
            )
        self.cached = self.prepare = self.identity = None
