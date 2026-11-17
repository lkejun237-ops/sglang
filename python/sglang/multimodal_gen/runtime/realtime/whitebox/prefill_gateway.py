# SPDX-License-Identifier: Apache-2.0
"""Authenticated, bounded prefill calls; no Room, CLI or recording ownership."""

import asyncio
import hashlib
import json
import logging
import re
import time
from uuid import uuid4

import msgspec.msgpack
from fastapi import HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.responses import Response

from .interactive import create_model_leases
from .rtc_auth import RTCCapabilities

logger = logging.getLogger(__name__)
MAX_PAYLOAD = 64 * 1024**2 + 4096
MAX_CHUNK = 1024**2
TERMINAL = {"persisted", "failed", "cancelled"}


class PrefillRateLimited(ValueError):
    pass


def terminal(record):
    return record["state"] in TERMINAL or (
        record["state"] == "memory_ready" and not record.get("durable", False)
    )


def bind_world(native, world):
    from ..cache_archive import world_token

    if not isinstance(world, dict) or set(world) != {"source_id", "world_id"}:
        raise ValueError("Invalid prefill world identity")
    for value in world.values():
        if not isinstance(value, str) or not re.fullmatch(
            r"[A-Za-z0-9._-]{1,128}", value
        ):
            raise ValueError("Invalid prefill world identity")
    world_token(world["world_id"])
    # Ownership comes from the signed envelope, never from native payload fields.
    return {**native, "world_id": world["world_id"], "cache_write": True}


class PrefillStatusCors:
    """CORS only for bearer-authorized status reads, never legacy routes."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith(
            "/v1/realtime_video/prefill/requests/"
        ):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        if scope["method"] == "OPTIONS":
            requested = {
                v.strip().lower()
                for v in headers.get(b"access-control-request-headers", b"").split(b",")
            }
            valid = (
                headers.get(b"access-control-request-method") == b"GET"
                and b"authorization" in requested
                and requested <= {b"authorization", b"content-type"}
            )
            response = Response(
                status_code=204 if valid else 403,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type",
                    "Cache-Control": "no-store",
                },
            )
            return await response(scope, receive, send)

        async def forward(message):
            if message["type"] == "http.response.start":
                response_headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower()
                    not in {
                        b"access-control-allow-origin",
                        b"access-control-allow-credentials",
                        b"cache-control",
                    }
                ]
                message = {
                    **message,
                    "headers": response_headers
                    + [
                        (b"access-control-allow-origin", b"*"),
                        (b"cache-control", b"no-store"),
                    ],
                }
            await send(message)

        await self.app(scope, receive, forward)


def request_key(user, world, request_id):
    if not isinstance(request_id, str) or not re.fullmatch(
        r"[A-Za-z0-9._-]{1,128}", request_id
    ):
        raise ValueError("Invalid prefill request identity")
    return hashlib.sha256(
        json.dumps([user, world, request_id], sort_keys=True).encode()
    ).hexdigest()


class MemoryPrefillStore:
    """Test/local store. Production uses a shared Redis result store."""

    def __init__(self):
        self.values = {}
        self.rate = {}

    async def get(self, key):
        value = self.values.get(key)
        return (
            None if value is None or value["expires_at"] <= time.time() else dict(value)
        )

    async def claim(self, key, value):
        old = await self.get(key)
        if old is not None:
            return False, old
        minute = int(time.time() // 60)
        self.rate = {k: v for k, v in self.rate.items() if k[0] == minute}
        subject = value.get("subject_hash", "local")
        for bucket, limit in (((minute, subject), 10), ((minute, "global"), 30)):
            if self.rate.get(bucket, 0) >= limit:
                raise PrefillRateLimited("Prefill request rate exceeded")
        self.rate[(minute, subject)] = self.rate.get((minute, subject), 0) + 1
        self.rate[(minute, "global")] = self.rate.get((minute, "global"), 0) + 1
        self.values[key] = dict(value)
        return True, dict(value)

    async def update(self, key, owner, value):
        current = await self.get(key)
        if current is None or current["owner"] != owner:
            raise RuntimeError("Prefill ownership changed")
        self.values[key] = dict(value)


class RedisPrefillStore:
    def __init__(self, client, *, prefix="whitebox:prefill:v1:"):
        self.client, self.prefix = client, prefix

    async def get(self, key):
        from redis.exceptions import ConnectionError, TimeoutError

        try:
            value = await self.client.get(self.prefix + key)
        except (ConnectionError, TimeoutError):
            # Managed Redis / proxies may close an idle pooled connection.
            # GET can safely reconnect once; never replay a claim/update whose
            # write outcome is unknown.
            value = await self.client.get(self.prefix + key)
        return None if value is None else json.loads(value)

    async def claim(self, key, value):
        minute = str(int(time.time() // 60))
        subject = value["subject_hash"]
        result = await self.client.eval(
            "local old=redis.call('GET',KEYS[1]); if old then return {0,old} end; "
            "if tonumber(redis.call('GET',KEYS[2]) or '0')>=10 or tonumber(redis.call('GET',KEYS[3]) or '0')>=30 then return {-1,''} end; "
            "redis.call('INCR',KEYS[2]); redis.call('EXPIRE',KEYS[2],70); "
            "redis.call('INCR',KEYS[3]); redis.call('EXPIRE',KEYS[3],70); "
            "redis.call('SET',KEYS[1],ARGV[1],'EX',3600); return {1,ARGV[1]}",
            3,
            self.prefix + key,
            self.prefix + "rate:" + subject + ":" + minute,
            self.prefix + "rate:global:" + minute,
            json.dumps(value),
        )
        if result[0] == -1:
            raise PrefillRateLimited("Prefill request rate exceeded")
        return result[0] == 1, json.loads(result[1])

    async def update(self, key, owner, value):
        encoded = json.dumps(value)
        if len(encoded.encode()) > 64 * 1024:
            raise ValueError("Prefill receipt exceeded its budget")
        result = await self.client.eval(
            "local v=redis.call('GET',KEYS[1]); if not v or cjson.decode(v).owner~=ARGV[1] then return 0 end; "
            "redis.call('SET',KEYS[1],ARGV[2],'EX',ARGV[3]); return 1",
            1,
            self.prefix + key,
            owner,
            encoded,
            max(1, int(value["expires_at"] - time.time())),
        )
        if result != 1:
            raise RuntimeError("Prefill ownership changed")


def public_result(record):
    result = {
        k: v
        for k, v in record.items()
        if k not in {"owner", "payload_sha256", "expires_at", "subject_hash"}
    }
    if not terminal(record) and time.time() - record["updated_at"] > 600:
        result.update(state="unknown", code="PREFILL_STATUS_STALE")
    return result


class PrefillGateway:
    def __init__(
        self, *, coordinator, connect, secret, model_revision, vae_fingerprint, store
    ):
        self.coordinator, self.connect, self.store = coordinator, connect, store
        self.capabilities = RTCCapabilities(secret, audience="whitebox-prefill")
        self.model_revision, self.vae_fingerprint = model_revision, vae_fingerprint
        self.tasks = {}
        self.uploads = 0
        self.batch_tasks = set()

    def authorize(self, token, *, origin, world, request_id, operation):
        claims = self.capabilities.verify(
            token,
            origin=origin,
            world=world,
            operation=operation,
            scope="world",
            consume=False,
        )
        if claims.get("request_id") != request_id:
            raise ValueError("Invalid prefill request identity")
        return claims

    async def _publish(self, key, record, state, **fields):
        record.update(state=state, updated_at=time.time(), **fields)
        await self.store.update(key, record["owner"], record)

    async def _run(self, key, record, native, user_id):
        began = time.monotonic()
        durable = record.get("durable", False)
        lease = create_model_leases(
            self.coordinator,
            self.connect,
            user_id=user_id,
            model_revision=self.model_revision,
            vae_fingerprint=self.vae_fingerprint,
            trace_id=record["trace_id"],
            max_bytes=MAX_PAYLOAD,
            strict_release=True,
            capacity_wait_ms=60_000 if record.get("batch") else 0,
            operation_path="/v1/realtime_video/prefill",
            profile_hint=native.get("size"),
        )
        try:
            async with asyncio.timeout(540):
                async with lease.open() as (upstream, assignment):
                    admission_ms = (time.monotonic() - began) * 1000
                    ready = msgspec.msgpack.decode(await upstream.recv())
                    if ready.get("type") != "prefill.ready" or (
                        durable and not ready.get("capabilities", {}).get("persistence")
                    ):
                        raise RuntimeError("Prefill capability is unavailable")
                    await self._publish(
                        key, record, "computing", admission_ms=admission_ms
                    )
                    await upstream.send(
                        msgspec.msgpack.encode({**native, "persist": durable})
                    )
                    memory_ready = False
                    while True:
                        message = msgspec.msgpack.decode(await upstream.recv())
                        if (
                            not isinstance(message, dict)
                            or message.get("request_id") != record["request_id"]
                        ):
                            raise RuntimeError("Prefill receipt identity mismatch")
                        kind = message.get("type")
                        # Native non-durable completion is a legacy L1-ready
                        # alias, never evidence of durable storage.
                        if kind == "prefill.completed":
                            kind = "prefill.memory_ready"
                            message = {**message, "type": kind}
                        if kind in {"prefill.memory_ready", "prefill.persisted"}:
                            if message.get("session_id") != assignment.session_id:
                                raise RuntimeError("Prefill session identity mismatch")
                        if kind == "prefill.memory_ready":
                            await lease.release_compute()
                            memory_ready = True
                            await self._publish(
                                key,
                                record,
                                "memory_ready",
                                receipt=message,
                                prefill_to_memory_ms=(time.monotonic() - began) * 1000,
                            )
                            if not durable:
                                return
                        elif kind == "prefill.persisting":
                            await self._publish(key, record, "persisting")
                        elif kind == "prefill.persisted":
                            if not memory_ready:
                                raise RuntimeError(
                                    "Prefill persistence preceded cleanup"
                                )
                            await self._publish(
                                key,
                                record,
                                "persisted",
                                receipt=message,
                                gateway_total_ms=(time.monotonic() - began) * 1000,
                            )
                            return
                        elif kind == "prefill.failed":
                            await self._publish(key, record, "failed", receipt=message)
                            return
                        else:
                            raise RuntimeError("Unexpected prefill receipt")
        except asyncio.CancelledError:
            await self._publish(
                key,
                record,
                "cancelled",
                code="PREFILL_CANCELLED",
                content="Prefill was cancelled.",
            )
            raise
        except Exception as exc:
            logger.exception("Prefill request failed trace_id=%s", record["trace_id"])
            busy = (getattr(exc, "reason", None) or getattr(exc, "code", None)) in {
                "CAPACITY_EXHAUSTED",
                "USER_SESSION_LIMIT",
                "USER_ALREADY_ACTIVE",
            }
            await self._publish(
                key,
                record,
                "failed",
                code="PREFILL_BUSY" if busy else "PREFILL_FAILED",
                content="The service is busy. Please try again shortly."
                if busy
                else "Prefill could not be completed.",
            )
        finally:
            self.tasks.pop(key, None)

    async def serve(self, ws):
        await ws.accept()
        key = task = None
        uploading = False
        try:
            first = await asyncio.wait_for(ws.receive_bytes(), 10)
            if len(first) > 8192:
                raise ValueError("Prefill authorization is too large")
            header = msgspec.msgpack.decode(first)
            if not isinstance(header, dict) or header.get("type") != "prefill.begin":
                raise ValueError("Prefill authorization is required")
            world, request_id = header.get("world"), header.get("request_id")
            if not isinstance(world, dict) or set(world) != {"source_id", "world_id"}:
                raise ValueError("Invalid prefill world identity")
            claims = self.authorize(
                header.get("authorization"),
                origin=ws.headers.get("origin"),
                world=world,
                request_id=request_id,
                operation="submit",
            )
            key = request_key(claims["sub"], world, request_id)
            size, digest = claims.get("payload_bytes"), claims.get("payload_sha256")
            if (
                type(size) is not int
                or not 1 <= size <= MAX_PAYLOAD
                or not isinstance(digest, str)
                or not re.fullmatch(r"[a-f0-9]{64}", digest)
            ):
                raise ValueError("Invalid prefill upload budget")
            old = await self.store.get(key)
            if old is None:
                if self.uploads >= 4 or len(self.tasks) >= 4:
                    raise ValueError("Prefill upload capacity is unavailable")
                self.uploads += 1
                uploading = True
                upload_started = time.monotonic()
                await ws.send_bytes(
                    msgspec.msgpack.encode(
                        dict(
                            type="prefill.upload_ready",
                            request_id=request_id,
                            max_chunk_bytes=MAX_CHUNK,
                        )
                    )
                )
                body = bytearray()
                async with asyncio.timeout(60):
                    while len(body) < size:
                        chunk = await ws.receive_bytes()
                        if (
                            not chunk
                            or len(chunk) > MAX_CHUNK
                            or len(body) + len(chunk) > size
                        ):
                            raise ValueError("Invalid prefill upload chunk")
                        body.extend(chunk)
                if hashlib.sha256(body).hexdigest() != digest:
                    raise ValueError("Prefill upload checksum mismatch")
                native = msgspec.msgpack.decode(body)
                if (
                    not isinstance(native, dict)
                    or native.get("type") != "prefill"
                    or native.get("request_id") != request_id
                ):
                    raise ValueError("Invalid prefill request")
                allowed = {
                    "type",
                    "request_id",
                    "prompt",
                    "seed",
                    "size",
                    "source_stream",
                    "source_frame",
                    "source_frame_size",
                    "references",
                    "zing_realtime_api_version",
                    "persist",
                    "persist_timeout_s",
                }
                if set(native) - allowed:
                    raise ValueError("Unsupported public prefill fields")
                if type(native.get("persist", False)) is not bool:
                    raise ValueError("Invalid prefill persistence option")
                native = bind_world(native, world)
                now = time.time()
                record = dict(
                    durable=native.get("persist", False),
                    request_id=request_id,
                    trace_id=uuid4().hex,
                    state="accepted",
                    owner=uuid4().hex,
                    payload_sha256=digest,
                    created_at=now,
                    subject_hash=hashlib.sha256(claims["sub"].encode()).hexdigest(),
                    updated_at=now,
                    expires_at=now + 3600,
                    upload_ms=(time.monotonic() - upload_started) * 1000,
                )
                if len(self.tasks) >= 4:
                    raise ValueError("Prefill execution capacity is unavailable")
                claimed, old = await self.store.claim(key, record)
                if claimed:
                    task = asyncio.create_task(
                        self._run(key, record, native, claims["sub"])
                    )
                    self.tasks[key] = task
                self.uploads -= 1
                uploading = False
                del body
            if old["payload_sha256"] != digest:
                raise ValueError("Prefill request ID already has different content")
            previous = None
            async with asyncio.timeout(550):
                while True:
                    result = await self.store.get(key)
                    if result is None:
                        raise RuntimeError("Prefill result expired")
                    public = public_result(result)
                    if public != previous:
                        await ws.send_bytes(
                            msgspec.msgpack.encode({"type": "prefill.status", **public})
                        )
                        previous = public
                    if terminal(public) or public["state"] == "unknown":
                        break
                    await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Prefill connection failed")
            try:
                await ws.send_bytes(
                    msgspec.msgpack.encode(
                        dict(
                            type="prefill.failed",
                            code="PREFILL_REQUEST_FAILED",
                            content="The prefill request could not be completed. Please verify its authorization and input.",
                        )
                    )
                )
            except Exception:
                pass
        finally:
            if uploading:
                self.uploads -= 1
            # Explicit API submission has a bounded lifetime independent of its
            # observer connection. Reconnect via status; never start another run.
            try:
                await ws.close()
            except Exception:
                pass

    async def status(self, request, request_id):
        world = dict(
            world_id=request.query_params.get("world_id"),
            source_id=request.query_params.get("source_id"),
        )
        try:
            authorization = request.headers.get("authorization", "")
            if not authorization.startswith("Bearer "):
                raise ValueError()
            claims = self.authorize(
                authorization[7:],
                origin=request.headers.get("origin"),
                world=world,
                request_id=request_id,
                operation="status",
            )
            result = await self.store.get(request_key(claims["sub"], world, request_id))
        except ValueError:
            raise HTTPException(
                403, "Prefill authorization could not be verified"
            ) from None
        if result is None:
            raise HTTPException(404, "The prefill request was not found")
        return public_result(result)

    async def _claim_batch_item(self, key, record):
        # Only a known atomic rate rejection is retried. Connection failures
        # have an unknown write outcome and must not replay a claim.
        began = time.monotonic()
        async with asyncio.timeout(125):
            while True:
                try:
                    record["batch_rate_wait_ms"] = (time.monotonic() - began) * 1000
                    return await self.store.claim(key, record)
                except PrefillRateLimited:
                    await asyncio.sleep(max(0.1, 60 - time.time() % 60 + 0.05))

    async def batch(self, request):
        """Wait for durable results, reusing per-item claims and model leases.

        Envelope: msgpack {items: [{world, authorization, payload: bytes}]}.
        Each payload is the existing signed native Prefill with persist=true.
        No extra SDK fields or second GPU scheduler are introduced.
        """
        if self.batch_tasks:
            raise HTTPException(429, "A prefill batch is already running")
        if (
            request.headers.get("content-type", "").split(";")[0]
            != "application/msgpack"
        ):
            raise HTTPException(415, "A MessagePack batch is required")
        body = bytearray()
        async with asyncio.timeout(60):
            async for chunk in request.stream():
                if len(body) + len(chunk) > MAX_PAYLOAD:
                    raise HTTPException(
                        413, "The prefill batch exceeds its byte budget"
                    )
                body.extend(chunk)
        try:
            envelope = msgspec.msgpack.decode(body)
            if not isinstance(envelope, dict) or set(envelope) != {"items"}:
                raise ValueError()
            items = envelope["items"]
            if not isinstance(items, list) or not 1 <= len(items) <= 8:
                raise ValueError()
            validated, seen = [], set()
            for item in items:
                if not isinstance(item, dict) or set(item) != {
                    "world",
                    "authorization",
                    "payload",
                }:
                    raise ValueError()
                raw = item["payload"]
                if not isinstance(raw, bytes):
                    raise ValueError()
                native = msgspec.msgpack.decode(raw)
                if (
                    not isinstance(native, dict)
                    or native.get("type") != "prefill"
                    or native.get("persist") is not True
                ):
                    raise ValueError()
                allowed = {
                    "type",
                    "request_id",
                    "prompt",
                    "seed",
                    "size",
                    "source_stream",
                    "source_frame",
                    "source_frame_size",
                    "references",
                    "zing_realtime_api_version",
                    "persist",
                    "persist_timeout_s",
                }
                if set(native) - allowed:
                    raise ValueError()
                request_id = native.get("request_id")
                claims = self.authorize(
                    item["authorization"],
                    origin=request.headers.get("origin"),
                    world=item["world"],
                    request_id=request_id,
                    operation="submit",
                )
                digest = hashlib.sha256(raw).hexdigest()
                if claims.get("payload_sha256") != digest or claims.get(
                    "payload_bytes"
                ) != len(raw):
                    raise ValueError()
                key = request_key(claims["sub"], item["world"], request_id)
                if key in seen:
                    raise ValueError()
                seen.add(key)
                validated.append(
                    (key, bind_world(native, item["world"]), claims["sub"], digest)
                )
        except (ValueError, TypeError, KeyError, msgspec.DecodeError):
            raise HTTPException(
                422, "The prefill batch authorization or input is invalid"
            ) from None
        if self.batch_tasks:
            raise HTTPException(429, "A prefill batch is already running")

        async def execute():
            results = []
            for key, native, user, digest in validated:
                now = time.time()
                record = dict(
                    request_id=native["request_id"],
                    trace_id=uuid4().hex,
                    durable=True,
                    batch=True,
                    state="accepted",
                    owner=uuid4().hex,
                    payload_sha256=digest,
                    created_at=now,
                    updated_at=now,
                    expires_at=now + 3600,
                    subject_hash=hashlib.sha256(user.encode()).hexdigest(),
                )
                try:
                    claimed, old = await self._claim_batch_item(key, record)
                except Exception:
                    results.append(
                        dict(
                            request_id=native["request_id"],
                            state="unknown",
                            code="PREFILL_SUBMISSION_UNCONFIRMED",
                        )
                    )
                    continue
                if old["payload_sha256"] != digest or not old.get("durable"):
                    results.append(
                        dict(
                            request_id=native["request_id"],
                            state="failed",
                            code="PREFILL_REQUEST_ID_CONFLICT",
                        )
                    )
                    continue
                try:
                    if claimed:
                        if len(self.tasks) >= 4:
                            await self._publish(
                                key, record, "failed", code="PREFILL_BUSY"
                            )
                        else:
                            task = asyncio.create_task(
                                self._run(key, record, native, user)
                            )
                            self.tasks[key] = task
                            await asyncio.shield(task)
                    else:
                        async with asyncio.timeout(550):
                            while not terminal(old):
                                await asyncio.sleep(0.25)
                                old = await self.store.get(key)
                                if (
                                    old is None
                                    or public_result(old)["state"] == "unknown"
                                ):
                                    break
                    result = await self.store.get(key)
                    results.append(
                        public_result(result)
                        if result
                        else {"request_id": native["request_id"], "state": "unknown"}
                    )
                except Exception:
                    results.append(
                        dict(
                            request_id=native["request_id"],
                            state="unknown",
                            code="PREFILL_STATUS_UNCONFIRMED",
                        )
                    )
            durable = sum(row["state"] == "persisted" for row in results)
            return dict(
                state="succeeded"
                if durable == len(results)
                else "partial"
                if durable
                else "unknown"
                if any(row["state"] == "unknown" for row in results)
                else "failed",
                durable_count=durable,
                items=results,
            )

        task = asyncio.create_task(execute())
        self.batch_tasks.add(task)

        def finished(task):
            self.batch_tasks.discard(task)
            if not task.cancelled():
                task.exception()

        task.add_done_callback(finished)
        # A disconnect does not create another compute attempt. Existing per-item
        # status reads/retry identities remain authoritative.
        return await asyncio.shield(task)

    async def close(self):
        tasks = [*self.batch_tasks, *self.tasks.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def register_prefill_routes(app, manager):
    @app.websocket("/v1/realtime_video/prefill")
    async def prefill(ws: WebSocket):
        await manager.serve(ws)

    @app.get("/v1/realtime_video/prefill/requests/{request_id}")
    async def status(request: Request, request_id: str):
        return await manager.status(request, request_id)

    @app.post("/v1/realtime_video/prefill/batch")
    async def batch(request: Request):
        return await manager.batch(request)
