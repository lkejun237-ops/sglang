# SPDX-License-Identifier: Apache-2.0
"""Local bound facade over independent Encoder input and Decoder output sockets."""

import asyncio
import base64
import time
from urllib.parse import urlsplit, urlunsplit

import msgspec

from .contracts import PoolError, canonical


class PooledBoundSocket:
    def __init__(
        self,
        control,
        assignment,
        *,
        connect,
        max_bytes,
        receipt_timeout=90,
        on_close=None,
        persistence_observer=None,
    ):
        self.control, self.assignment, self.connect = control, assignment, connect
        self.max_bytes, self.receipt_timeout = max_bytes, receipt_timeout
        self.queue = asyncio.Queue(maxsize=16)
        self.active = None
        self.closed = False
        self.close_task = None
        self.on_close = on_close
        self.persistence_observer = persistence_observer
        self.background = set()
        self.queue.put_nowait(
            msgspec.msgpack.encode(
                dict(
                    type="bound.ready",
                    protocol="bound-compute-v1",
                    worker_epoch=assignment["boot_epochs"]["dit"],
                    player_expires_at=assignment["player_expires_at"],
                )
            )
        )

    async def emit(self, message):
        if not self.closed:
            await self.queue.put(msgspec.msgpack.encode(message))

    async def recv(self):
        value = await self.queue.get()
        if isinstance(value, BaseException):
            raise value
        return value

    def identity(self):
        return dict(
            admission_id=self.assignment["intent"]["admission_id"],
            allocation_id=self.assignment["allocation_id"],
        )

    async def authorization(self, role, child):
        result = await self.control(
            "business",
            {
                **self.identity(),
                "role": role,
                "session_id": child["session_id"],
                "generation_id": child["generation_id"],
            },
        )
        return {
            "Authorization": "Bearer " + result["token"],
            "X-Role-Pool-Claim": base64.urlsafe_b64encode(
                canonical(result["body"]).encode()
            )
            .decode()
            .rstrip("="),
        }

    async def send(self, wire):
        if self.closed:
            raise PoolError("MODEL_CONNECTION_CLOSED")
        value = msgspec.msgpack.decode(wire)
        kind = value.get("type")
        if kind == "bound.open":
            if self.active is not None:
                raise PoolError("COMPUTE_ALREADY_ACTIVE")
            if value.get("kind") not in {"generate", "prefill"}:
                raise PoolError("INVALID_OPERATION")
            child = dict(
                value,
                input=None,
                output=None,
                task=None,
                finish=None,
                accepted=0,
                consumed=0,
                pending_credit=None,
                input_done=False,
                output_done=False,
                submitted=0,
                submitted_bytes=0,
                boundaries={},
                received=0,
                accepted_event=asyncio.Event(),
                persist_requested=False,
            )
            self.active = child
            url = self.assignment["topology"]["encoder"]["endpoints"]["input_ws"]
            if child["kind"] == "prefill":
                p = urlsplit(url)
                url = urlunsplit(
                    (p.scheme, p.netloc, "/v1/realtime_video/prefill", "", "")
                )
            try:
                child["input"] = await self.connect(
                    url,
                    additional_headers=await self.authorization("encoder", child),
                    max_size=self.max_bytes,
                    max_queue=1,
                    compression=None,
                    close_timeout=2,
                    open_timeout=10,
                )
            except BaseException:
                self.active = None
                raise
            await self.emit(
                dict(type="bound.opened", operation_id=child["operation_id"])
            )
            child["task"] = asyncio.create_task(self._read_input(child))
        elif (
            self.active is None
            or value.get("operation_id") != self.active["operation_id"]
        ):
            raise PoolError("OPERATION_IDENTITY_MISMATCH")
        elif kind == "bound.data":
            child = self.active
            payload = msgspec.msgpack.decode(value["payload"])
            if payload.get("type") == "prefill":
                child["persist_requested"] = payload.get("persist") is True
            if payload.get("kind") == "source.append":
                frames = payload["payload"]
                if frames["source_frame_start"] != child["submitted"]:
                    raise PoolError("INVALID_SOURCE_POSITION")
                child["submitted"] += frames["num_frames"]
                child["submitted_bytes"] += len(frames["data"])
                child["boundaries"][child["submitted"]] = child["submitted_bytes"]
            await child["input"].send(value["payload"])
        elif kind == "bound.cancel":
            await self._finish(self.active)
        else:
            raise PoolError("INVALID_BOUND_MESSAGE")

    async def _data(self, child, value, *, locked=False):
        if not locked:
            async with self._wire_lock(child):
                return await self._data(child, value, locked=True)
        wire = value if isinstance(value, bytes) else msgspec.msgpack.encode(value)
        await self.emit(
            dict(type="bound.data", operation_id=child["operation_id"], payload=wire)
        )

    async def _read_input(self, child):
        try:
            async for wire in child["input"]:
                value = msgspec.msgpack.decode(wire)
                kind = value.get("type")
                if kind == "prefill.ready":
                    continue
                if kind == "session.ready":
                    if any(
                        value.get(k) != child[k]
                        for k in ("session_id", "generation_id")
                    ):
                        raise PoolError("MODEL_SESSION_MISMATCH")
                    url = self.assignment["topology"]["decoder"]["endpoints"][
                        "output_ws"
                    ]
                    output = child["output"] = await self.connect(
                        url,
                        additional_headers=await self.authorization("decoder", child),
                        max_size=self.max_bytes,
                        max_queue=1,
                        compression=None,
                        close_timeout=2,
                        open_timeout=10,
                    )
                    await output.send(
                        msgspec.msgpack.encode(
                            dict(
                                type="output.attach",
                                session_id=child["session_id"],
                                generation_id=child["generation_id"],
                                token=value["output_attachment"]["token"],
                            )
                        )
                    )
                    ready = msgspec.msgpack.decode(await output.recv())
                    if ready != dict(
                        type="output.ready",
                        session_id=child["session_id"],
                        generation_id=child["generation_id"],
                    ):
                        raise PoolError("OUTPUT_ATTACHMENT_MISMATCH")
                    child["output_task"] = asyncio.create_task(self._read_output(child))
                    await self._data(child, value)
                elif kind == "source.accepted":
                    end = value["source_frame_end"]
                    if (
                        value["source_frame_start"] != child["accepted"]
                        or not child["accepted"] < end <= child["submitted"]
                    ):
                        raise PoolError("INVALID_SOURCE_ACK")
                    child["accepted"] = end
                    child["accepted_event"].set()
                    await self._data(child, value)
                    await self._credit(child)
                elif kind in {"input.completed", "session.completed"}:
                    child["input_done"] = True
                    await self._complete(child)
                elif kind in {"prefill.completed", "prefill.memory_ready"}:
                    descriptor = value.pop("_role_persistence_descriptor", None)
                    value["type"] = "prefill.memory_ready"
                    await self._data(child, value)
                    if descriptor is not None:
                        if self.persistence_observer is not None:
                            self.persistence_observer(value, descriptor)
                        else:
                            task = asyncio.create_task(
                                self._persistence(value, descriptor)
                            )
                            self.background.add(task)
                            task.add_done_callback(self.background.discard)
                    elif child["persist_requested"]:
                        await self.emit(
                            dict(
                                type="prefill.failed",
                                request_id=value["request_id"],
                                memory_ready=True,
                                persistence_state="failed",
                                code="CACHE_PERSISTENCE_UNAVAILABLE",
                                content="Prepared cache storage is unavailable.",
                                fatal=False,
                            )
                        )
                    asyncio.create_task(self._finish(child))
                    return
                elif kind == "source.moments":
                    continue
                else:
                    await self._data(child, value)
        except asyncio.CancelledError:
            raise
        except BaseException:
            asyncio.create_task(self._finish(child, code="MODEL_OPERATION_FAILED"))
        else:
            if not child["input_done"] and child["finish"] is None:
                asyncio.create_task(self._finish(child, code="MODEL_OPERATION_FAILED"))

    async def _credit(self, child):
        value = child["pending_credit"]
        if value and value["source_frame_end"] <= child["accepted"]:
            child["consumed"] = value["source_frame_end"]
            child["pending_credit"] = None
            await self._data(child, value)
            for end in list(child["boundaries"]):
                if end < child["consumed"]:
                    del child["boundaries"][end]

    async def _frame_ready(self, child, value):
        start, count = value["output_frame_start"], value["num_frames"]
        if start != child["received"] or count < 1:
            raise PoolError("INVALID_OUTPUT_POSITION")
        end = start + count
        while child["accepted"] < end:
            child["accepted_event"].clear()
            if child["accepted"] < end:
                await child["accepted_event"].wait()
        child["received"] = end

    async def _read_output(self, child):
        try:
            async for wire in child["output"]:
                value = msgspec.msgpack.decode(wire)
                kind = value.get("type")
                if kind == "source.consumed":
                    previous = (child["pending_credit"] or {}).get(
                        "source_frame_end", child["consumed"]
                    )
                    if (
                        value["source_frame_end"] < previous
                        or child["boundaries"].get(value["source_frame_end"])
                        != value["source_bytes_end"]
                    ):
                        raise PoolError("INVALID_SOURCE_CREDIT")
                    child["pending_credit"] = value
                    await self._credit(child)
                elif kind == "session.completed":
                    child["output_done"] = True
                    child["output_completion"] = value
                    await self._complete(child)
                    return
                elif kind == "frame_batch_header":
                    # Header and raw payload stay adjacent on the logical bound
                    # stream; no Encoder ACK may be inserted between them.
                    payload = await child["output"].recv()
                    await self._frame_ready(child, value)
                    async with self._wire_lock(child):
                        await self._data(child, wire, locked=True)
                        await self._data(child, payload, locked=True)
                else:
                    if kind == "frame_batch":
                        await self._frame_ready(child, value)
                    async with self._wire_lock(child):
                        await self._data(child, wire, locked=True)
        except asyncio.CancelledError:
            raise
        except BaseException:
            asyncio.create_task(self._finish(child, code="MODEL_OPERATION_FAILED"))

        else:
            if not child["output_done"] and child["finish"] is None:
                asyncio.create_task(self._finish(child, code="MODEL_OPERATION_FAILED"))

    def _wire_lock(self, child):
        if "wire_lock" not in child:
            child["wire_lock"] = asyncio.Lock()
        return child["wire_lock"]

    async def _complete(self, child):
        if child["input_done"] and child["output_done"]:
            await self._credit(child)
            if (
                child["pending_credit"] is not None
                or len(
                    {
                        child[k]
                        for k in ("submitted", "accepted", "consumed", "received")
                    }
                )
                != 1
            ):
                asyncio.create_task(self._finish(child, code="MODEL_OPERATION_FAILED"))
            else:
                await self._data(child, child["output_completion"])
                asyncio.create_task(self._finish(child))

    async def _finish(self, child, code=None):
        if child["finish"] is not None:
            return await asyncio.shield(child["finish"])

        async def cleanup():
            cleaned = False
            try:
                for key in ("input", "output"):
                    if child[key] is not None:
                        await child[key].close()
                for key in ("task", "output_task"):
                    task = child.get(key)
                    if task is not None and task is not asyncio.current_task():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                deadline = time.monotonic() + self.receipt_timeout
                while time.monotonic() < deadline:
                    status = await self.control(
                        "child-status",
                        {**self.identity(), "session_id": child["session_id"]},
                    )
                    receipt = status.get("child_receipts", {}).get(child["session_id"])
                    if (
                        receipt
                        and receipt.get("generation_id") == child["generation_id"]
                        and receipt.get("cleaned") is True
                    ):
                        cleaned = True
                        break
                    await asyncio.sleep(0.05)
            except Exception:
                pass
            finally:
                if self.active is child:
                    self.active = None
                await self.emit(
                    dict(
                        type="bound.completed",
                        operation_id=child["operation_id"],
                        cleaned=cleaned,
                        **(
                            {"code": code or "RELEASE_UNKNOWN"}
                            if code or not cleaned
                            else {}
                        ),
                    )
                )
                if not cleaned:
                    self.closed = True

        child["finish"] = asyncio.create_task(cleanup())
        return await asyncio.shield(child["finish"])

    async def _persistence(self, receipt, descriptor):
        await observe_prefill_persistence(
            self.control,
            self.identity(),
            receipt,
            descriptor,
            emit=self.emit,
            closed=lambda: self.closed,
        )

    async def close(self):
        if self.close_task is None:
            if self.on_close is not None:
                self.on_close()

            async def close_io():
                self.closed = True
                child = self.active
                tasks = list(self.background)
                sockets = []
                if child is not None:
                    tasks.extend(
                        t
                        for t in (
                            child.get("task"),
                            child.get("output_task"),
                            child.get("finish"),
                        )
                        if t is not None
                    )
                    sockets = [
                        child[k] for k in ("input", "output") if child[k] is not None
                    ]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(
                    *(socket.close() for socket in sockets), return_exceptions=True
                )
                await asyncio.gather(*tasks, return_exceptions=True)
                self.active = None
                while not self.queue.empty():
                    self.queue.get_nowait()
                self.queue.put_nowait(PoolError("MODEL_CONNECTION_CLOSED"))

            self.close_task = asyncio.create_task(close_io())
        await asyncio.shield(self.close_task)


async def observe_prefill_persistence(
    control, identity, receipt, descriptor, *, emit, closed
):
    """Observe only the exact writeback; never renew a released group."""
    deadline = time.monotonic() + 180
    try:
        while not closed() and time.monotonic() < deadline:
            value = await control("persistence", {**identity, "descriptor": descriptor})
            if value.get("state") == "persisted":
                await emit({**receipt, "type": "prefill.persisted", "storage": value})
                return
            if value.get("state") == "failed":
                raise PoolError("PREFILL_PERSISTENCE_FAILED")
            await asyncio.sleep(0.5)
        if not closed():
            raise PoolError("PREFILL_PERSISTENCE_TIMEOUT")
    except asyncio.CancelledError:
        raise
    except Exception:
        await emit(
            dict(
                type="prefill.failed",
                request_id=receipt["request_id"],
                memory_ready=True,
                persistence_state="unknown",
                code="PREFILL_PERSISTENCE_UNCONFIRMED",
                content="Saving the prepared cache has not been confirmed.",
                fatal=False,
            )
        )
