# SPDX-License-Identifier: Apache-2.0
"""Authenticated conditioning RPC with an optional single task prefix cache."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from types import SimpleNamespace

from .cache import ConditioningCache, tensor_bytes
from .contract import ConditionError, _tensor_spec, canonical_json, safe_json
from .prefix_task import SinglePrefixTask
from .resources import AdmissionLedger, BoundedExecutor, ResourceLimits
from .wire import decode_request, pack_message

logger = logging.getLogger(__name__)
PREFIX = "/v1/conditioning"


class _Disconnected(Exception):
    pass


@dataclass
class _Flight:
    key: tuple
    value: object
    future: object
    enqueued_at: float
    waiters: int = 1
    state: str = "queued"
    encode_ms: float = 0
    queue_ms: float = 0
    retained_reservations: list = field(default_factory=list)


class EncodeQueue:
    """FIFO queue whose last queued waiter can leave without submitting work."""

    def __init__(
        self,
        encoder,
        cache,
        limits,
        flights,
        on_encoded,
        clock=time.monotonic,
        *,
        keep_device=False,
    ):
        self.keep_device = keep_device
        self.encoder, self.cache, self.limits = encoder, cache, limits
        self.flights, self.on_encoded, self.clock = flights, on_encoded, clock
        self.pending = deque()
        self.wake = asyncio.Event()
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="conditioning-encoder"
        )
        self.runner = None
        self.closed = False

    @property
    def depth(self):
        return sum(f.state == "queued" and f.waiters > 0 for f in self.pending)

    def enqueue(self, key, value):
        if self.closed:
            raise ConditionError("CONDITION_UNAVAILABLE")
        if self.depth >= self.limits.max_queued_encodes:
            raise ConditionError("CONDITION_BUSY")
        future = asyncio.get_running_loop().create_future()
        # A running encode can finish after its last waiter has disconnected.
        # Consume otherwise-unobserved exceptions without changing await behavior.
        future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        flight = _Flight(key, value, future, self.clock())
        self.flights[key] = flight
        self.pending.append(flight)
        self.wake.set()
        if self.runner is None:
            self.runner = asyncio.create_task(self._run())
        return flight

    def leave(self, flight):
        flight.waiters -= 1
        if flight.waiters == 0 and flight.state == "queued":
            flight.state = "discarded"
            if self.flights.get(flight.key) is flight:
                del self.flights[flight.key]
            flight.future.cancel()
            # Remove immediately so abandoned payloads do not occupy an unbounded
            # deque while an uninterruptible native forward is running.
            try:
                self.pending.remove(flight)
            except ValueError:
                pass

    def _forward(self, value, key):
        import torch

        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        with (
            torch.inference_mode(),
            set_forward_context(current_timestep=0, attn_metadata=None),
        ):
            task = getattr(value, "task", None)
            if task is not None and task.closed:
                raise ConditionError("CONDITION_TASK_EXPIRED")
            if self.keep_device:
                condition = self.encoder.encode_device(value)
            else:
                condition = self.encoder(value)
                condition = {
                    name: tensor.detach().to("cpu").contiguous()
                    for name, tensor in condition.items()
                }
            if task is not None and task.closed:
                raise ConditionError("CONDITION_TASK_EXPIRED")
            _tensor_spec(condition, self.limits.max_condition_rows)
            if tensor_bytes(condition) > self.limits.max_condition_bytes:
                raise ConditionError(
                    "CONDITION_INVALID", "Condition exceeds the output byte budget"
                )
            if task is None:
                self.cache.put(key, condition)
            return condition

    async def _run(self):
        while True:
            if not self.pending:
                if self.closed:
                    return
                self.wake.clear()
                await self.wake.wait()
                continue
            flight = self.pending.popleft()
            if flight.state != "queued" or flight.waiters == 0:
                continue
            flight.state = "running"
            started = self.clock()
            flight.queue_ms = max(0, (started - flight.enqueued_at) * 1000)
            future = asyncio.get_running_loop().run_in_executor(
                self.pool, self._forward, flight.value, flight.key
            )
            try:
                condition = await asyncio.shield(future)
                flight.encode_ms = max(0, (self.clock() - started) * 1000)
                self.on_encoded(flight)
                flight.future.set_result(condition)
            except Exception as error:
                if not isinstance(error, ConditionError):
                    logger.exception("Conditioning encoder failed")
                    error = ConditionError("CONDITION_UNAVAILABLE")
                flight.future.set_exception(error)
            finally:
                flight.state = "done"
                flight.value = None
                for reservation in flight.retained_reservations:
                    reservation.release()
                flight.retained_reservations.clear()
                if self.flights.get(flight.key) is flight:
                    del self.flights[flight.key]

                # Requests own the published result until delivery completes.
                # The idle queue must not pin its last CUDA result (including
                # through either Future) while waiting for the next request.
                condition = future = flight = None

    async def close(self):
        self.closed = True
        for flight in list(self.pending):
            flight.state = "discarded"
            if not flight.future.done():
                flight.future.set_exception(ConditionError("CONDITION_UNAVAILABLE"))
            self.flights.pop(flight.key, None)
        self.pending.clear()
        self.wake.set()
        if self.runner:
            # Do not cancel a running encoder. Native GPU work must drain before
            # its request buffers and this executor can be released.
            while not self.runner.done():
                try:
                    await asyncio.shield(self.runner)
                except asyncio.CancelledError:
                    continue
            self.runner.result()
        self.pool.shutdown(wait=True, cancel_futures=False)


class ConditioningService:
    def __init__(
        self,
        *,
        profile,
        encoder,
        credentials=None,
        tokens=None,
        limits=None,
        clock=time.monotonic,
        result_transport="http",
    ):
        if credentials is not None and tokens is not None:
            raise ValueError("Configure conditioning tokens only once")
        self.profile, self.encoder = profile, encoder
        self.credentials = dict(tokens if tokens is not None else credentials or {})
        if not self.credentials:
            raise ValueError("Conditioning service credentials are required")
        for token, principal in self.credentials.items():
            if (
                type(token) is not str
                or not token
                or not token.isascii()
                or type(principal) is not dict
                or type(principal.get("tenant")) is not str
                or not principal["tenant"]
                or type(principal.get("admin", False)) is not bool
            ):
                raise ValueError("Invalid conditioning token principal")
        self.limits, self.clock = limits or ResourceLimits(), clock
        self.ledger = AdmissionLedger(self.limits.budgets())
        self.cpu = BoundedExecutor(
            self.limits.cpu_executor_active, self.limits.cpu_executor_waiting
        )
        self._cpu_gate = asyncio.Semaphore(self.cpu.capacity)
        if result_transport not in {"http", "nccl"}:
            raise ValueError("Unknown conditioning result transport")
        self.nccl = None
        if result_transport == "nccl":
            from .nccl import NcclSender

            if not callable(getattr(encoder, "encode_device", None)):
                raise ValueError("NCCL conditioning requires a device-native encoder")
            self.nccl = NcclSender(
                f"cuda:{encoder.gpu_id}",
                self.limits.request_timeout_s,
                max_bridges=getattr(self.limits, "max_bridges", 8),
            )
        self.cache = ConditioningCache(
            0 if self.nccl else self.limits.cache_bytes,
            per_tenant_bytes=0 if self.nccl else self.limits.cache_bytes_per_tenant,
        )
        self.flights = {}
        self._stats = defaultdict(self._new_stats)
        self.queue = EncodeQueue(
            encoder,
            self.cache,
            self.limits,
            self.flights,
            self._encoded,
            clock,
            keep_device=self.nccl is not None,
        )
        self.prefix_tasks = (
            SinglePrefixTask(
                lambda: encoder.create_prefix_cache(self.limits.prefix_cache_bytes),
                self.queue.pool,
                self.limits.prefix_task_idle_s,
                clock,
            )
            if hasattr(encoder, "create_prefix_cache")
            else None
        )
        self._closed = False
        self._requests = set()
        self._close_task = None

    @staticmethod
    def _new_stats():
        return dict(
            hit=0,
            miss=0,
            joined=0,
            admission_rejected=0,
            response_bytes=0,
            encode_ms=deque(maxlen=1024),
            queue_ms=deque(maxlen=1024),
        )

    def _encoded(self, flight):
        stats = self._stats[flight.key[0]]
        stats["encode_ms"].append(flight.encode_ms)
        stats["queue_ms"].append(flight.queue_ms)

    def _authenticate(self, headers):
        if sum(len(k) + len(v) + 4 for k, v in headers) > self.limits.max_header_bytes:
            raise ConditionError()
        values = {}
        for key, value in headers:
            key = key.lower()
            if key in values:
                raise ConditionError()
            values[key] = value
        authorization = values.get(b"authorization", b"")
        if not authorization.startswith(b"Bearer "):
            raise ConditionError("CONDITION_UNAUTHORIZED")
        try:
            token = authorization[7:].decode("ascii")
        except UnicodeError:
            raise ConditionError("CONDITION_UNAUTHORIZED") from None
        principal = self.credentials.get(token)
        if not principal:
            raise ConditionError("CONDITION_UNAUTHORIZED")
        return principal, values

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await self.close()
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        principal = None
        response_started = False
        raw_send = send

        async def tracked_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await raw_send(message)

        send = tracked_send
        task = asyncio.current_task()
        try:
            principal, headers = self._authenticate(scope.get("headers", []))
            if self._closed:
                raise ConditionError("CONDITION_UNAVAILABLE")
            path, method = scope["path"], scope["method"]
            if path == PREFIX + "/profile" and method == "GET":
                return await self._json(
                    send,
                    200,
                    dict(
                        schema_version=1,
                        result_transport="nccl" if self.nccl else "http",
                        profile_id=self.profile.profile_id,
                        preprocess_version=self.profile.document["preprocess_version"],
                        task_prefix_cache_v1=self.prefix_tasks is not None,
                        tensor_contract={
                            "context_dtype": "BF16",
                            "width": 5120,
                            "row_role_dtype": "I64",
                        },
                    ),
                )
            if path == PREFIX + "/stats" and method == "GET":
                return await self._json(send, 200, self._stats_snapshot(principal))
            if self.prefix_tasks is not None and path.startswith(PREFIX + "/tasks/"):
                if method in ("POST", "DELETE"):
                    self._requests.add(task)
                    return await self._task_request(
                        principal["tenant"],
                        method,
                        path.removeprefix(PREFIX + "/tasks/"),
                        receive,
                        send,
                    )

            if self.nccl is not None and path in {
                PREFIX + "/encode",
                PREFIX + "/bridge/open",
                PREFIX + "/bridge/close",
            }:
                peer = scope.get("client")
                if peer is None or peer[0] not in {"127.0.0.1", "::1"}:
                    raise ConditionError("CONDITION_UNAUTHORIZED")
            if (
                path == PREFIX + "/bridge/open"
                and method == "POST"
                and self.nccl is not None
            ):
                self._requests.add(task)
                return await self._open_bridge(
                    principal["tenant"], headers, receive, send
                )
            if (
                path == PREFIX + "/bridge/close"
                and method == "POST"
                and self.nccl is not None
            ):
                self._requests.add(task)
                return await self._close_bridge(
                    principal["tenant"], headers, receive, send
                )
            if path != PREFIX + "/encode" or method != "POST":
                raise ConditionError("CONDITION_INVALID", 404)
            self._requests.add(task)
            if self.nccl is not None:
                bridge_id = headers.get(b"x-conditioning-bridge", b"").decode("ascii")
                self.nccl.acquire(
                    bridge_id,
                    principal["tenant"],
                )
                try:
                    await self._encode(
                        principal["tenant"],
                        headers,
                        receive,
                        send,
                        bridge_id=bridge_id,
                    )
                finally:
                    self.nccl.release(bridge_id)
            else:
                await self._encode(principal["tenant"], headers, receive, send)
        except _Disconnected:
            return
        except ConditionError as error:
            if response_started:
                raise
            if error.code == "CONDITION_BUSY" and principal:
                self._stats[principal["tenant"]]["admission_rejected"] += 1
            await self._json(
                send,
                error.status,
                {"code": error.code, "message": error.public_message},
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            if response_started:
                raise
            logger.exception("Conditioning RPC failed")
            error = ConditionError("CONDITION_UNAVAILABLE")
            await self._json(
                send,
                error.status,
                {"code": error.code, "message": error.public_message},
            )
        finally:
            self._requests.discard(task)

    async def _task_request(self, tenant, method, task_id, receive, send):
        reservations = [self.ledger.reserve(tenant, slots=1)]
        deadline = self.clock() + self.limits.request_timeout_s
        watcher = releasing = None
        try:
            try:
                await asyncio.wait_for(
                    self._read_body(receive, 0, tenant, reservations),
                    max(0, deadline - self.clock()),
                )
            except asyncio.TimeoutError:
                raise _Disconnected() from None
            watcher = asyncio.create_task(self._disconnect(receive))
            if method == "POST":
                response = {"task_handle": self.prefix_tasks.open(tenant, task_id)}
            else:
                releasing = asyncio.create_task(
                    self.prefix_tasks.release(tenant, task_id)
                )
                done, _ = await asyncio.wait(
                    [releasing, watcher],
                    timeout=max(0, deadline - self.clock()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if watcher in done:
                    watcher.result()
                if releasing not in done:
                    raise _Disconnected()
                releasing.result()
                response = {"released": True}
            await self._send_response(
                send,
                canonical_json(response),
                watcher,
                deadline,
                content_type=b"application/json",
            )
        except ConditionError as error:
            await self._send_response(
                send,
                canonical_json({"code": error.code, "message": error.public_message}),
                watcher,
                deadline,
                status=error.status,
                content_type=b"application/json",
            )
        finally:
            try:
                if releasing is not None:
                    # SinglePrefixTask.release shields the underlying drain.
                    await self._cancel_and_drain(releasing)
                if watcher is not None:
                    await self._cancel_and_drain(watcher)
            finally:
                for reservation in reversed(reservations):
                    reservation.release()

    async def _open_bridge(self, tenant, headers, receive, send):
        size = self._body_length(headers)
        if size > 1024:
            raise ConditionError("CONDITION_INVALID")
        reservations = [self.ledger.reserve(tenant, slots=1)]
        try:
            body = await asyncio.wait_for(
                self._read_body(receive, size, tenant, reservations),
                self.limits.request_timeout_s,
            )
            doc = safe_json(body)
            await self.nccl.open(doc, tenant, self._cpu_run)
            opened_id = (
                doc["bridge_id"]
                if isinstance(doc, dict) and "bridge_id" in doc
                else self.nccl.bridge_id
            )
            await self._json(send, 200, {"bridge_id": opened_id})
        finally:
            for reservation in reversed(reservations):
                reservation.release()

    async def _close_bridge(self, tenant, headers, receive, send):
        size = self._body_length(headers)
        if size > 1024:
            raise ConditionError("CONDITION_INVALID")
        reservations = [self.ledger.reserve(tenant, slots=1)]
        try:
            body = await asyncio.wait_for(
                self._read_body(receive, size, tenant, reservations),
                self.limits.request_timeout_s,
            )
            doc = safe_json(body)
            if (
                type(doc) is not dict
                or "bridge_id" not in doc
                or set(doc) - {"bridge_id", "force"}
                or ("force" in doc and type(doc["force"]) is not bool)
            ):
                raise ConditionError(
                    "CONDITION_INVALID", "Invalid NCCL bridge close request"
                )
            result = self.nccl.close_bridge(
                doc["bridge_id"], tenant, force=bool(doc.get("force", False))
            )
            await self._json(send, 200, result)
        finally:
            for reservation in reversed(reservations):
                reservation.release()

    def _invoke_nccl_send(self, condition, bridge_id):
        params = inspect.signature(self.nccl.send).parameters
        if len(params) >= 2:
            return self.nccl.send(condition, bridge_id)
        return self.nccl.send(condition)

    async def _send_nccl(
        self, send, condition, header, watcher, deadline, *, bridge_id=None
    ):
        if self.clock() >= deadline:
            raise ConditionError("CONDITION_TIMEOUT")
        if watcher.done():
            watcher.result()
            raise _Disconnected()
        active_bridge_id = bridge_id or self.nccl.bridge_id
        metadata = (
            canonical_json(
                dict(
                    schema_version=1,
                    profile_id=self.profile.profile_id,
                    input_digest=header["input_digest"],
                    bridge_id=active_bridge_id,
                    rows=condition["context"].shape[0],
                )
            )
            + b"\n"
        )
        completion = canonical_json({"done": header["input_digest"]}) + b"\n"

        async def stream():
            try:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"application/x-ndjson")],
                    }
                )
                # Receiver must know shape and allocate before NCCL send waits.
                await send(
                    {"type": "http.response.body", "body": metadata, "more_body": True}
                )
                await self._cpu_run(self._invoke_nccl_send, condition, active_bridge_id)
                await send(
                    {
                        "type": "http.response.body",
                        "body": completion,
                        "more_body": False,
                    }
                )
                return len(metadata) + len(completion)
            except BaseException as error:
                self.nccl.fail(active_bridge_id)
                if isinstance(error, ConditionError):
                    # Headers already belong to this streaming result. The
                    # outer encode error handler must not send a second reply.
                    raise RuntimeError(
                        "Conditioning NCCL result transfer failed"
                    ) from error
                raise

        sender = asyncio.create_task(stream())
        try:
            done, _ = await asyncio.wait(
                [sender, watcher],
                timeout=max(0, deadline - self.clock()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if sender in done:
                return sender.result()
            raise _Disconnected()
        finally:
            # BoundedExecutor drains an in-flight GPU operation on cancellation;
            # result/ledger ownership must last through that drain.
            await self._cancel_and_drain(sender)

    async def _cpu_run(self, function, *args):
        # Requests waiting outside the executor remain bounded by request slots.
        # This avoids spurious BUSY responses for a legal singleflight burst.
        async with self._cpu_gate:
            return await self.cpu.run(function, *args)

    def _body_length(self, headers):
        value = headers.get(b"content-length")
        if (
            b"transfer-encoding" in headers
            or value is None
            or not value
            or not value.isdigit()
        ):
            raise ConditionError()
        if (
            headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
            != b"application/octet-stream"
        ):
            raise ConditionError()
        try:
            size = int(value)
        except ValueError:
            raise ConditionError() from None
        if size > min(
            self.limits.max_request_bytes,
            self.limits.max_input_body_bytes,
        ):
            raise ConditionError("CONDITION_INVALID", 413)
        if size < 4:
            raise ConditionError()
        return size

    async def _read_body(self, receive, expected, tenant, reservations):
        body, actual = bytearray(), 0
        charge = self.ledger.reserve(tenant, inflight=0)
        reservations.append(charge)
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise _Disconnected()
            if message["type"] != "http.request":
                raise ConditionError()
            chunk = message.get("body", b"")
            if type(chunk) is not bytes:
                raise ConditionError()
            actual += len(chunk)
            if actual > expected:
                raise ConditionError()
            if chunk:
                charge.grow(inflight=len(chunk))
                body.extend(chunk)
            if not message.get("more_body", False):
                if actual != expected:
                    raise ConditionError()
                return bytes(body)

    @staticmethod
    async def _disconnect(receive):
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            # ASGI forbids extra request chunks after more_body=False. Treat
            # extra transport data as an invalid request, without busy looping.
            raise ConditionError("CONDITION_INVALID", "Unexpected ASGI request event")

    async def _encode(self, tenant, headers, receive, send, *, bridge_id=None):
        expected = self._body_length(headers)
        reservations, flight, watcher = [], None, None
        prefix_task = None
        left_flight = False
        begun = self.clock()
        try:
            reservations.append(self.ledger.reserve(tenant, slots=1))
            try:
                body = await asyncio.wait_for(
                    self._read_body(receive, expected, tenant, reservations),
                    self.limits.request_timeout_s,
                )
                decoded = await asyncio.wait_for(
                    self._cpu_run(decode_request, body, self.profile, self.limits),
                    max(0.001, self.limits.request_timeout_s - (self.clock() - begun)),
                )
            except asyncio.TimeoutError:
                raise ConditionError("CONDITION_TIMEOUT") from None
            header, media = decoded.header, decoded.media
            budget = self.limits.request_timeout_s
            # Compare whole seconds before true division: huge positive JSON
            # integers must clamp without float conversion, and multiplying an
            # otherwise finite configured timeout by 1000 could also overflow.
            if header["deadline_ms"] // 1000 < budget:
                budget = min(budget, header["deadline_ms"] / 1000)
            deadline = begun + budget
            if self.clock() >= deadline:
                raise ConditionError("CONDITION_TIMEOUT")
            if "task_handle" in header:
                if self.prefix_tasks is None:
                    raise ConditionError("CONDITION_INVALID")
                prefix_task = self.prefix_tasks.acquire(tenant, header["task_handle"])
            key = (
                (tenant, header["input_digest"])
                if prefix_task is None
                else (tenant, prefix_task.handle, header["input_digest"])
            )
            cached_size = self.cache.entry_size(key) if prefix_task is None else None
            status, encode_ms, queue_ms = "hit", 0, 0
            output_charge = (
                cached_size
                if cached_size is not None
                else self.limits.output_reservation_bytes
            )
            output_reservation = self.ledger.reserve(tenant, inflight=output_charge)
            reservations.append(output_reservation)
            condition = (
                await self._cpu_run(self.cache.get, key)
                if cached_size is not None
                else None
            )
            # CPU lookup may outlive the budget, including an LRU eviction while
            # it waits. Stop before funding a miss or joining/starting native work.
            if self.clock() >= deadline:
                raise ConditionError("CONDITION_TIMEOUT")
            if prefix_task is not None and prefix_task.closed:
                raise ConditionError("CONDITION_TASK_EXPIRED")
            if (
                condition is None
                and output_charge < self.limits.output_reservation_bytes
            ):
                # The LRU may evict the entry while this request waits for a CPU
                # thread. Reserve the miss bound before enqueueing any work.
                output_reservation.grow(
                    inflight=self.limits.output_reservation_bytes - output_charge
                )
            watcher = asyncio.create_task(self._disconnect(receive))
            if condition is None:
                flight = self.flights.get(key)
                if flight is None:
                    status = "miss"
                    flight = self.queue.enqueue(
                        key,
                        SimpleNamespace(
                            prompt=header["prompt"],
                            media=media,
                            media_identity=decoded.media_identity,
                            task=prefix_task,
                        ),
                    )
                else:
                    status = "joined"
                    flight.waiters += 1
                self._stats[tenant][status] += 1
                waiting = asyncio.shield(flight.future)
                try:
                    done, _ = await asyncio.wait(
                        [waiting, watcher],
                        timeout=max(0, deadline - self.clock()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if watcher in done:
                        watcher.result()
                        raise _Disconnected()
                    if waiting not in done:
                        raise ConditionError("CONDITION_TIMEOUT")
                    condition = waiting.result()
                finally:
                    waiting.cancel()
                    # Encoding ownership ends when waiting ends, including before
                    # an error response whose transport may stall. Response
                    # reservations still belong to this request until send ends.
                    self.queue.leave(flight)
                    left_flight = True
                encode_ms, queue_ms = flight.encode_ms, flight.queue_ms
                output_reservation.reduce(inflight=tensor_bytes(condition))
            else:
                self._stats[tenant]["hit"] += 1
            if prefix_task is not None and prefix_task.closed:
                raise ConditionError("CONDITION_TASK_EXPIRED")
            if self.nccl is not None:
                control_bytes = await self._send_nccl(
                    send,
                    condition,
                    header,
                    watcher,
                    deadline,
                    bridge_id=bridge_id,
                )
                self._stats[tenant]["response_bytes"] += control_bytes
                return
            response = await self._cpu_run(
                pack_message,
                dict(
                    schema_version=1,
                    profile_id=self.profile.profile_id,
                    input_digest=header["input_digest"],
                    cache=status,
                    encode_ms=encode_ms,
                    queue_ms=queue_ms,
                    tensors=["context", "row_role"],
                ),
                condition,
            )
            if self.clock() >= deadline:
                raise ConditionError("CONDITION_TIMEOUT")
            if prefix_task is not None and prefix_task.closed:
                raise ConditionError("CONDITION_TASK_EXPIRED")
            if len(response) > self.limits.max_response_bytes:
                raise ConditionError(
                    "CONDITION_INVALID", "RPC response exceeds the byte budget"
                )
            if watcher.done():
                watcher.result()
                raise _Disconnected()
            # Keep both slot and body/result bytes charged until the last send
            # completes. Headers/framing are covered by the bounded CPU executor.
            await self._send_response(send, response, watcher, deadline)
            self._stats[tenant]["response_bytes"] += len(response)
        except ConditionError as error:
            if error.code == "CONDITION_BUSY":
                self._stats[tenant]["admission_rejected"] += 1
            # Error responses retain the same request slot/body ownership until
            # their final send. A short transmission grace allows an expired
            # deadline to report TIMEOUT, without an indefinitely blocked sender.
            data = canonical_json({"code": error.code, "message": error.public_message})
            await self._send_response(
                send,
                data,
                watcher,
                self.clock() + min(1.0, self.limits.request_timeout_s),
                status=error.status,
                content_type=b"application/json",
            )
        finally:
            if prefix_task is not None:
                self.prefix_tasks.leave(prefix_task)
            if flight is not None:
                if flight.state == "running" and flight.waiters == (
                    0 if left_flight else 1
                ):
                    # The last request may stop waiting, but native work still
                    # owns its input and worst-case result buffers. Transfer the
                    # existing byte charge without growing or double-charging it.
                    for reservation in reservations:
                        size = reservation.charges.get("inflight", 0)
                        if size:
                            flight.retained_reservations.append(
                                reservation.split(inflight=size)
                            )
                if not left_flight:
                    self.queue.leave(flight)
            try:
                if watcher:
                    await self._cancel_and_drain(watcher)
            finally:
                # Repeated cancellation of cleanup must never skip ledger release.
                for reservation in reversed(reservations):
                    reservation.release()

    async def _send_response(
        self,
        send,
        response,
        watcher,
        deadline,
        *,
        status=200,
        content_type=b"application/octet-stream",
    ):
        if status == 200 and self.clock() >= deadline:
            raise ConditionError("CONDITION_TIMEOUT")
        sender = asyncio.create_task(
            self._response(send, status, response, content_type)
        )
        try:
            done, _ = await asyncio.wait(
                ([sender, watcher] if watcher is not None else [sender]),
                timeout=max(0, deadline - self.clock()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if sender in done:
                return sender.result()
            if watcher is not None and watcher in done:
                watcher.result()
            # Once response headers are sent a JSON error cannot replace them.
            # Cancel the transport send and end ownership when it disconnects or
            # stalls beyond the same overall request deadline.
            raise _Disconnected()
        finally:
            await self._cancel_and_drain(sender)

    @staticmethod
    async def _cancel_and_drain(task):
        if not task.done():
            task.cancel()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # Cancellation of the owner must not abandon a live transport
                # send or its buffers. The first cancellation still propagates
                # through the calling finally after this terminal cleanup.
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()

    def _stats_snapshot(self, principal):
        selected = (
            list(self._stats.values())
            if principal.get("admin")
            else [self._stats[principal["tenant"]]]
        )
        result = {
            name: sum(stats[name] for stats in selected)
            for name in (
                "hit",
                "miss",
                "joined",
                "admission_rejected",
                "response_bytes",
            )
        }
        # Tenant stats reveal only that tenant's queued work.
        tenant = None if principal.get("admin") else principal["tenant"]
        # RPC admission/singleflight owns request hit/miss accounting. The LRU
        # contributes capacity/eviction data, scoped by the same principal.
        result["cache"] = {
            name: value
            for name, value in self.cache.snapshot(tenant=tenant).items()
            if name not in {"hit", "miss", "joined"}
        }
        result["queue_depth"] = sum(
            f.state == "queued"
            and f.waiters > 0
            and (tenant is None or f.key[0] == tenant)
            for f in self.queue.pending
        )
        for name in ("encode_ms", "queue_ms"):
            values = sorted(v for stats in selected for v in stats[name])
            result[name] = {
                label: values[min(len(values) - 1, int((len(values) - 1) * fraction))]
                if values
                else 0
                for label, fraction in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
            }
        return result

    @staticmethod
    async def _response(send, status, body, content_type):
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", content_type),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def _json(self, send, status, value):
        await self._response(send, status, canonical_json(value), b"application/json")

    async def _close(self):
        requests = list(self._requests)
        for task in requests:
            task.cancel()
        await asyncio.gather(*requests, return_exceptions=True)
        if self.prefix_tasks is not None:
            await self.prefix_tasks.close()

        if self.nccl is not None and hasattr(self.nccl, "close"):
            self.nccl.close()
        await self.queue.close()
        await self.cpu.close()

    async def close(self):
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                continue
        self._close_task.result()
