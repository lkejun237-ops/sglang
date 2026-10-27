# SPDX-License-Identifier: Apache-2.0
"""Single-consumer, same-node Qwen result channel.

HTTP carries input/identity/shape and the terminal acknowledgement. NCCL carries
only the two CUDA tensors. A partial exchange poisons the channel: retrying on
the same communicator could pair an old send with a new prompt's receive.
"""

import asyncio
import concurrent.futures
import threading
import uuid
from datetime import timedelta

import httpx
import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.tensor_bridge import (
    bridge_group,
    receive_into,
    send_tensor,
)

from .client import _network_loop
from .contract import ConditionError, canonical_json, safe_json

PREFIX = "/v1/conditioning"
GROUP_PREFIX = "zing-conditioning-bridge"
NAMES = ("context", "row_role")


def result_shapes(
    metadata, *, profile_id, input_digest, bridge_id, max_rows, max_bytes
):
    """Validate bounded metadata before allocating or entering a collective."""
    if (
        type(metadata) is not dict
        or metadata.get("schema_version") != 1
        or metadata.get("profile_id") != profile_id
        or metadata.get("input_digest") != input_digest
        or metadata.get("bridge_id") != bridge_id
    ):
        raise ConditionError("CONDITION_INVALID", "Invalid NCCL result identity")
    rows = metadata.get("rows")
    if type(rows) is not int or not 0 < rows <= max_rows or rows * 10248 > max_bytes:
        raise ConditionError("CONDITION_INVALID", "Invalid NCCL result capacity")
    return {"context": (rows, 5120), "row_role": (rows,)}


class _Records:
    """Bounded newline JSON reader, independent of HTTP chunk boundaries."""

    def __init__(self, chunks):
        self.chunks, self.buffer = chunks.__aiter__(), bytearray()
        self.total = 0

    async def read(self):
        while b"\n" not in self.buffer:
            try:
                chunk = await self.chunks.__anext__()
            except StopAsyncIteration:
                raise ConditionError(
                    "CONDITION_UNAVAILABLE", "Incomplete NCCL reply"
                ) from None
            self.total += len(chunk)
            if self.total > 65536:
                raise ConditionError(
                    "CONDITION_INVALID", "NCCL reply exceeds metadata limit"
                )
            self.buffer.extend(chunk)
        line, _, remaining = self.buffer.partition(b"\n")
        self.buffer = bytearray(remaining)
        return safe_json(bytes(line))

    async def finish(self):
        if self.buffer:
            raise ConditionError("CONDITION_INVALID", "Unexpected NCCL reply data")
        async for chunk in self.chunks:
            if chunk:
                raise ConditionError("CONDITION_INVALID", "Unexpected NCCL reply data")


class NcclSender:
    """Service-owned persistent group; busy is owned by the ASGI event loop."""

    def __init__(self, device, timeout_s):
        self.device = torch.device(device)
        self.timeout_s = timeout_s
        self.group = self.store = self.stream = None
        self.bridge_id = self.tenant = None
        self.busy = self.failed = False
        self.pending = None

    async def open(self, document, tenant, run):
        if self.bridge_id is not None or self.busy or self.failed:
            raise ConditionError("CONDITION_BUSY")
        if (
            type(document) is not dict
            or set(document) != {"port", "bridge_id", "timeout_s"}
            or type(document["port"]) is not int
            or not 0 < document["port"] < 65536
            or type(document["bridge_id"]) is not str
            or len(document["bridge_id"]) != 32
            or any(c not in "0123456789abcdef" for c in document["bridge_id"])
            or type(document["timeout_s"]) not in (int, float)
            or not 0 < document["timeout_s"] <= self.timeout_s
        ):
            raise ConditionError("CONDITION_INVALID", "Invalid NCCL bridge handshake")
        self.busy = True
        self.bridge_id, self.tenant = document["bridge_id"], tenant
        try:
            await run(self._connect, document)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.busy = False

    def _connect(self, document):
        torch.cuda.set_device(self.device)
        timeout = document["timeout_s"]
        self.store = dist.TCPStore(
            "127.0.0.1", document["port"], 2, False, timedelta(seconds=timeout)
        )
        self.group = bridge_group(
            self.store, 1, prefix=GROUP_PREFIX, timeout_seconds=timeout
        )
        self.stream = torch.cuda.Stream(device=self.device)

    def acquire(self, bridge_id, tenant):
        if self.failed or self.group is None:
            raise ConditionError("CONDITION_UNAVAILABLE")
        if bridge_id != self.bridge_id or tenant != self.tenant:
            raise ConditionError("CONDITION_UNAUTHORIZED")
        if self.busy:
            raise ConditionError("CONDITION_BUSY")
        self.busy = True

    def send(self, condition):
        # Native encode_device drains its producer device before publishing the
        # result, so this private stream may consume it from an executor thread.
        torch.cuda.set_device(self.device)
        self.pending = condition
        with torch.cuda.stream(self.stream):
            for name in NAMES:
                value = condition[name]
                if value.device != self.device or not value.is_contiguous():
                    raise ConditionError(
                        "CONDITION_INVALID", "NCCL result must stay on the encoder GPU"
                    )
                send_tensor(self.group, value, peer=0)
            # Keep condition alive until NCCL no longer reads the allocation.
            self.stream.synchronize()
        self.pending = None


class NcclReceiver:
    """DiT leader-owned channel, independent of source encoder communication."""

    def __init__(self, client):
        self.client = client
        self.group = self.store = self.stream = self.device = None
        self.bridge_id = uuid.uuid4().hex
        self.failed = False
        self.pending = None
        self.lock = threading.Lock()

    def _remaining(self, deadline):
        remaining = deadline - self.client.clock()
        if remaining <= 0:
            raise ConditionError("CONDITION_TIMEOUT")
        return remaining

    def _wait(self, future, deadline):
        try:
            return future.result(timeout=self._remaining(deadline))
        except (TimeoutError, concurrent.futures.TimeoutError):
            raise ConditionError("CONDITION_TIMEOUT") from None

    def _open(self, device, deadline):
        client = self.client
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ConditionError(
                "CONDITION_INVALID", "NCCL conditioning requires a CUDA device"
            )
        torch.cuda.set_device(self.device)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.store = dist.TCPStore(
            "127.0.0.1",
            0,
            2,
            True,
            timedelta(seconds=self._remaining(deadline)),
            wait_for_workers=False,
        )
        body = canonical_json(
            dict(
                port=self.store.port,
                bridge_id=self.bridge_id,
                timeout_s=client.timeout_s,
            )
        )
        # Both participants must join concurrently. Do not use the retrying RPC
        # helper for a stateful open or a tensor exchange.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                client.transport,
                "POST",
                PREFIX + "/bridge/open",
                body,
                self._remaining(deadline),
            )
            self.group = bridge_group(
                self.store, 0, prefix=GROUP_PREFIX, timeout_seconds=client.timeout_s
            )
            response = self._wait(future, deadline)
            if response.status != 200 or safe_json(response.body) != {
                "bridge_id": self.bridge_id
            }:
                raise ConditionError(
                    "CONDITION_UNAVAILABLE", "NCCL bridge handshake failed"
                )
        self.stream = torch.cuda.Stream(device=self.device)

    def encode(self, body, input_digest, device):
        if not self.lock.acquire(blocking=False):
            raise ConditionError("CONDITION_BUSY")
        try:
            deadline = self.client.clock() + self.client.timeout_s
            if self.failed:
                raise ConditionError(
                    "CONDITION_UNAVAILABLE", "NCCL channel requires a worker restart"
                )
            if self.group is None:
                self._open(device, deadline)
            requested = torch.device(device)
            if requested.type != "cuda" or requested.index not in (
                None,
                self.device.index,
            ):
                raise ConditionError(
                    "CONDITION_INVALID", "NCCL receiver device changed"
                )
            self._remaining(deadline)
            return self._exchange(body, input_digest, deadline)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.lock.release()

    def _exchange(self, body, input_digest, deadline):
        metadata = concurrent.futures.Future()
        # Only HTTP runs on the shared I/O loop. CUDA work remains on the
        # calling thread; a network deadline must never release live buffers.
        completion = _network_loop().submit(
            lambda: self._control(body, input_digest, metadata, deadline)
        )
        shapes = result_shapes(
            self._wait(metadata, deadline),
            profile_id=self.client.profile_id,
            input_digest=input_digest,
            bridge_id=self.bridge_id,
            max_rows=self.client.max_rows,
            max_bytes=self.client.max_bytes,
        )
        self._remaining(deadline)
        torch.cuda.set_device(self.device)
        with torch.cuda.stream(self.stream):
            # Allocate both before receiving. Failed storage remains owned by
            # the poisoned channel until process exit, including after timeout.
            self.pending = {
                name: torch.empty(
                    shapes[name],
                    dtype=torch.bfloat16 if name == "context" else torch.int64,
                    device=self.device,
                )
                for name in NAMES
            }
            for name in NAMES:
                self._remaining(deadline)
                receive_into(self.group, self.pending[name], peer=1)
            self.stream.synchronize()
        self._wait(completion, deadline)
        self._remaining(deadline)
        condition, self.pending = self.pending, None
        return condition

    async def _control(self, body, input_digest, metadata, deadline):
        try:
            remaining = self._remaining(deadline)
            return await asyncio.wait_for(
                self._read_control(body, input_digest, metadata, deadline),
                timeout=remaining,
            )
        except (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException):
            error = ConditionError("CONDITION_TIMEOUT")
            if not metadata.done():
                metadata.set_exception(error)
            raise error from None
        except BaseException as error:
            if not metadata.done():
                metadata.set_exception(error)
            raise

    async def _read_control(self, body, input_digest, metadata, deadline):
        client = self.client
        headers = {
            "Authorization": "Bearer " + client.token,
            "Content-Type": "application/octet-stream",
            "X-Conditioning-Bridge": self.bridge_id,
        }
        async with httpx.AsyncClient(
            timeout=self._remaining(deadline), trust_env=False, follow_redirects=False
        ) as http:
            async with http.stream(
                "POST", client.url + PREFIX + "/encode", content=body, headers=headers
            ) as response:
                if response.status_code != 200:
                    raise ConditionError(
                        "CONDITION_UNAVAILABLE", "NCCL encode request rejected"
                    )
                records = _Records(response.aiter_raw())
                first = await records.read()
                self._remaining(deadline)
                metadata.set_result(first)
                if await records.read() != {"done": input_digest}:
                    raise ConditionError(
                        "CONDITION_INVALID", "Invalid NCCL completion identity"
                    )
                await records.finish()
