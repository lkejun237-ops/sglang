# SPDX-License-Identifier: Apache-2.0
"""Single-consumer, same-node Qwen result channel.

HTTP carries input/identity/shape and the terminal acknowledgement. NCCL carries
only the two CUDA tensors. A partial exchange poisons the channel: retrying on
the same communicator could pair an old send with a new prompt's receive.
"""

import asyncio
import concurrent.futures
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta

import httpx
import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.tensor_bridge import (
    bridge_group,
    receive_into,
    send_tensor,
)
from sglang.multimodal_gen.runtime.utils.zing_capture_stream import (
    conditioning_nccl_stream,
)

from .client import _network_loop
from .contract import ConditionError, canonical_json, safe_json

PREFIX = "/v1/conditioning"
GROUP_PREFIX = "zing-conditioning-bridge"
NAMES = ("context", "row_role")


class _TaskExpiredBeforeTransfer(ConditionError):
    """HTTP 410 before metadata: neither peer has started tensor transfer."""

    def __init__(self):
        super().__init__("CONDITION_TASK_EXPIRED")


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


@dataclass
class _BridgeSlot:
    bridge_id: str
    tenant: str | None = None
    group: object = None
    store: object = None
    stream: object = None
    busy: bool = False
    failed: bool = False
    pending: object = None
    last_used: float = 0.0


class NcclSender:
    """Service-owned persistent bridge pool; busy is owned by the ASGI event loop."""

    def __init__(
        self,
        device,
        timeout_s,
        *,
        max_bridges: int = 8,
        idle_ttl_s: float = 300.0,
        clock=time.monotonic,
    ):
        self.device = torch.device(device)
        self.timeout_s = timeout_s
        self.max_bridges = max_bridges
        self.idle_ttl_s = idle_ttl_s
        self.clock = clock
        self.bridges: dict[str, _BridgeSlot] = {}
        self._default_bridge_id: str | None = None
        self._opening: bool = False
        self._failed: bool = False

    def _legacy_slot(self, create: bool = False) -> _BridgeSlot | None:
        if self._default_bridge_id and self._default_bridge_id in self.bridges:
            return self.bridges[self._default_bridge_id]
        if self.bridges:
            return next(iter(self.bridges.values()))
        if create:
            slot = _BridgeSlot(bridge_id="")
            self.bridges[""] = slot
            self._default_bridge_id = ""
            return slot
        return None

    def register_bridge_for_test(
        self, bridge_id: str, tenant: str, *, group=None, stream=None
    ) -> _BridgeSlot:
        slot = _BridgeSlot(
            bridge_id=bridge_id,
            tenant=tenant,
            group=group,
            stream=stream,
            last_used=self.clock(),
        )
        self.bridges[bridge_id] = slot
        self._default_bridge_id = bridge_id
        return slot

    def _destroy_slot_group(self, slot: _BridgeSlot) -> None:
        grp = slot.group
        slot.group = None
        slot.store = None
        slot.stream = None
        if grp is not None:
            try:
                dist.destroy_process_group(grp)
            except Exception:
                pass

    def evict_stale_bridges(self) -> list[str]:
        """Evict failed or TTL-expired non-busy bridges to reclaim bridge slots."""
        now = self.clock()
        evicted: list[str] = []
        for bid, slot in list(self.bridges.items()):
            if slot.busy:
                continue
            is_expired = (
                self.idle_ttl_s is not None
                and self.idle_ttl_s > 0
                and slot.last_used > 0
                and (now - slot.last_used) >= self.idle_ttl_s
            )
            if slot.failed or is_expired:
                self.bridges.pop(bid, None)
                self._destroy_slot_group(slot)
                evicted.append(bid)
                if self._default_bridge_id == bid:
                    self._default_bridge_id = next(iter(self.bridges), None)
        return evicted

    def close_bridge(
        self, bridge_id: str, tenant: str | None = None, *, force: bool = False
    ) -> dict:
        """Idempotently close and remove a bridge slot, destroying its process group."""
        if (
            type(bridge_id) is not str
            or len(bridge_id) != 32
            or any(c not in "0123456789abcdef" for c in bridge_id)
        ):
            raise ConditionError("CONDITION_INVALID", "Invalid NCCL bridge identifier")
        slot = self.bridges.get(bridge_id)
        if slot is None:
            return {
                "bridge_id": bridge_id,
                "closed": False,
                "active_bridges": len(self.bridges),
            }
        if tenant is not None and slot.tenant is not None and tenant != slot.tenant:
            raise ConditionError("CONDITION_UNAUTHORIZED")
        if slot.busy and not force:
            raise ConditionError("CONDITION_BUSY")
        self.bridges.pop(bridge_id, None)
        if self._default_bridge_id == bridge_id:
            self._default_bridge_id = next(iter(self.bridges), None)
        self._destroy_slot_group(slot)
        return {
            "bridge_id": bridge_id,
            "closed": True,
            "active_bridges": len(self.bridges),
        }

    def close(self) -> None:
        """Destroy all bridge process groups and clear the bridge table."""
        for bid, slot in list(self.bridges.items()):
            self.bridges.pop(bid, None)
            self._destroy_slot_group(slot)
        self._default_bridge_id = None
        self._opening = False

    @property
    def bridge_id(self) -> str | None:
        slot = self._legacy_slot(create=False)
        return slot.bridge_id if slot and slot.bridge_id else None

    @bridge_id.setter
    def bridge_id(self, value: str | None) -> None:
        if value is None:
            self.bridges.clear()
            self._default_bridge_id = None
            return
        slot = self._legacy_slot(create=True)
        old_key = slot.bridge_id
        if old_key in self.bridges and old_key != value:
            del self.bridges[old_key]
        slot.bridge_id = value
        self.bridges[value] = slot
        self._default_bridge_id = value

    @property
    def tenant(self) -> str | None:
        slot = self._legacy_slot(create=False)
        return slot.tenant if slot else None

    @tenant.setter
    def tenant(self, value: str | None) -> None:
        slot = self._legacy_slot(create=True)
        slot.tenant = value

    @property
    def group(self):
        slot = self._legacy_slot(create=False)
        return slot.group if slot else None

    @group.setter
    def group(self, value) -> None:
        slot = self._legacy_slot(create=True)
        slot.group = value

    @property
    def store(self):
        slot = self._legacy_slot(create=False)
        return slot.store if slot else None

    @store.setter
    def store(self, value) -> None:
        slot = self._legacy_slot(create=True)
        slot.store = value

    @property
    def stream(self):
        slot = self._legacy_slot(create=False)
        return slot.stream if slot else None

    @stream.setter
    def stream(self, value) -> None:
        slot = self._legacy_slot(create=True)
        slot.stream = value

    @property
    def busy(self) -> bool:
        if self._opening:
            return True
        return any(s.busy for s in self.bridges.values())

    @busy.setter
    def busy(self, value: bool) -> None:
        self._opening = False
        for s in self.bridges.values():
            s.busy = bool(value)
        if value and not self.bridges:
            self._legacy_slot(create=True).busy = True

    @property
    def failed(self) -> bool:
        slot = self._legacy_slot(create=False)
        return (slot.failed if slot else False) or self._failed

    @failed.setter
    def failed(self, value: bool) -> None:
        self._failed = bool(value)
        slot = self._legacy_slot(create=True)
        slot.failed = bool(value)

    @property
    def pending(self):
        slot = self._legacy_slot(create=False)
        return slot.pending if slot else None

    @pending.setter
    def pending(self, value) -> None:
        slot = self._legacy_slot(create=True)
        slot.pending = value

    async def open(self, document, tenant, run):
        if (
            "" in self.bridges
            and self.bridges[""].group is None
            and not self.bridges[""].bridge_id
        ):
            del self.bridges[""]
        if len(self.bridges) >= self.max_bridges:
            self.evict_stale_bridges()
        if self._opening or len(self.bridges) >= self.max_bridges:
            raise ConditionError("CONDITION_BUSY")
        if (
            type(document) is not dict
            or not {"port", "bridge_id", "timeout_s"} <= set(document)
            or set(document) - {"port", "bridge_id", "timeout_s", "connect_timeout_s"}
            or type(document["port"]) is not int
            or not 0 < document["port"] < 65536
            or type(document["bridge_id"]) is not str
            or len(document["bridge_id"]) != 32
            or any(c not in "0123456789abcdef" for c in document["bridge_id"])
            or type(document["timeout_s"]) not in (int, float)
            or not 0 < document["timeout_s"] <= self.timeout_s
        ):
            raise ConditionError("CONDITION_INVALID", "Invalid NCCL bridge handshake")
        connect_timeout = document.get("connect_timeout_s", document["timeout_s"])
        if (
            type(connect_timeout) not in (int, float)
            or not 0 < connect_timeout <= document["timeout_s"]
        ):
            raise ConditionError("CONDITION_INVALID", "Invalid NCCL connection budget")
        bridge_id = document["bridge_id"]
        if bridge_id in self.bridges:
            raise ConditionError("CONDITION_BUSY")
        slot = _BridgeSlot(
            bridge_id=bridge_id,
            tenant=tenant,
            busy=True,
            last_used=self.clock(),
        )
        self.bridges[bridge_id] = slot
        self._default_bridge_id = bridge_id
        self._opening = True
        self._failed = False
        try:
            await run(self._connect, document, time.monotonic() + connect_timeout, slot)
        except BaseException:
            slot.failed = True
            self._failed = True
            self.bridges.pop(bridge_id, None)
            if self._default_bridge_id == bridge_id:
                self._default_bridge_id = next(iter(self.bridges), None)
            raise
        finally:
            slot.busy = False
            self._opening = False
        return {"ok": True, "bridge_id": bridge_id}

    def _connect(
        self, document, deadline: float | None = None, slot: _BridgeSlot | None = None
    ):
        if isinstance(deadline, _BridgeSlot) and slot is None:
            slot = deadline
            deadline = None
        target = slot or self._legacy_slot(create=True)
        torch.cuda.set_device(self.device)
        timeout = document["timeout_s"]
        if deadline is None:
            deadline = time.monotonic() + timeout
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ConditionError("CONDITION_TIMEOUT")
        target.store = dist.TCPStore(
            "127.0.0.1", document["port"], 2, False, timedelta(seconds=remaining)
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ConditionError("CONDITION_TIMEOUT")
        if hasattr(target.store, "set_timeout"):
            target.store.set_timeout(timedelta(seconds=remaining))
        target.group = bridge_group(
            target.store, 1, prefix=GROUP_PREFIX, timeout_seconds=timeout
        )
        if time.monotonic() >= deadline:
            raise ConditionError("CONDITION_TIMEOUT")
        if hasattr(target.store, "set_timeout"):
            target.store.set_timeout(timedelta(seconds=timeout))
        target.stream = torch.cuda.Stream(device=self.device)

    def acquire(self, bridge_id, tenant):
        if not self.bridges:
            raise ConditionError("CONDITION_UNAVAILABLE")
        slot = self.bridges.get(bridge_id)
        if slot is None:
            # When the sole bridge is already poisoned, report CONDITION_UNAVAILABLE
            # before tenant/bridge_id mismatch to preserve single-bridge failure semantics.
            if len(self.bridges) == 1:
                only_slot = next(iter(self.bridges.values()))
                if only_slot.failed or only_slot.group is None:
                    raise ConditionError("CONDITION_UNAVAILABLE")
            raise ConditionError("CONDITION_UNAUTHORIZED")
        if slot.failed or slot.group is None:
            raise ConditionError("CONDITION_UNAVAILABLE")
        if tenant != slot.tenant:
            raise ConditionError("CONDITION_UNAUTHORIZED")
        if slot.busy:
            raise ConditionError("CONDITION_BUSY")
        slot.busy = True
        slot.last_used = self.clock()
        return slot

    def release(self, bridge_id: str | None = None) -> None:
        slot = self.bridges.get(bridge_id) if bridge_id else self._legacy_slot(False)
        if slot is not None:
            slot.busy = False
            slot.last_used = self.clock()

    def fail(self, bridge_id: str | None = None) -> None:
        slot = self.bridges.get(bridge_id) if bridge_id else self._legacy_slot(False)
        if slot is not None:
            slot.failed = True

    def send(self, condition, bridge_id: str | None = None):
        # Native encode_device drains its producer device before publishing the
        # result, so this private stream may consume it from an executor thread.
        slot = self.bridges.get(bridge_id) if bridge_id else self._legacy_slot(False)
        if slot is None or slot.group is None:
            raise ConditionError("CONDITION_UNAVAILABLE")
        torch.cuda.set_device(self.device)
        slot.pending = condition
        with torch.cuda.stream(slot.stream):
            for name in NAMES:
                value = condition[name]
                if value.device != self.device or not value.is_contiguous():
                    raise ConditionError(
                        "CONDITION_INVALID", "NCCL result must stay on the encoder GPU"
                    )
                send_tensor(slot.group, value, peer=0)
            # Keep condition alive until NCCL no longer reads the allocation.
            slot.stream.synchronize()
        slot.pending = None


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
                connect_timeout_s=self._remaining(deadline),
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
            self.store.set_timeout(timedelta(seconds=self._remaining(deadline)))
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
        self._remaining(deadline)
        # The group/Store survive this request; lazy NCCL initialization and
        # later exchanges retain the normal operation timeout, not a short
        # first request's remaining connection budget.
        self.store.set_timeout(timedelta(seconds=client.timeout_s))
        self.stream = conditioning_nccl_stream(self.device)

    def encode(self, body, input_digest, device, *, deadline=None):
        deadline = (
            min(deadline, self.client.clock() + self.client.timeout_s)
            if deadline is not None
            else self.client.clock() + self.client.timeout_s
        )
        # A boundary fallback may have spent its budget waiting for prefetch or
        # serializing input. No exchange has started, so keep the channel usable.
        self._remaining(deadline)
        if not self.lock.acquire(blocking=False):
            raise ConditionError("CONDITION_BUSY")
        try:
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
        except _TaskExpiredBeforeTransfer:
            # The service rejected the lease before emitting any NCCL metadata.
            # Only this explicit pre-transfer response permits lease recovery.
            raise
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
                if response.status_code == 410:
                    raise _TaskExpiredBeforeTransfer()
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

    def close(self, *, timeout=None, force: bool = False):
        """Close the remote bridge slot on the conditioning service and destroy local PG."""
        with self.lock:
            grp = self.group
            self.group = None
            self.store = None
            self.stream = None
            result = {"bridge_id": self.bridge_id, "closed": False}
            if grp is not None:
                budget = (
                    timeout
                    if timeout is not None
                    else min(5.0, float(self.client.timeout_s))
                )
                body = canonical_json(
                    {"bridge_id": self.bridge_id, "force": bool(force)}
                )
                try:
                    response = self.client.transport(
                        "POST", PREFIX + "/bridge/close", body, budget
                    )
                    if response.status == 200:
                        result = safe_json(response.body)
                except Exception:
                    pass
                try:
                    dist.destroy_process_group(grp)
                except Exception:
                    pass
            return result
