import asyncio
import base64
import hashlib
import hmac
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

import msgspec.msgpack

from sglang.multimodal_gen.runtime.realtime.whitebox import prefill_gateway as gateway

SECRET = "prefill-test-secret-" + "x" * 32
WORLD = {"source_id": "test", "world_id": "world-one"}


def test_idle_redis_read_reconnects_once_without_replaying_writes():
    import pytest
    from redis.exceptions import ConnectionError

    class Client:
        def __init__(self, failures):
            self.failures, self.reads, self.writes = failures, 0, 0

        async def get(self, key):
            self.reads += 1
            if self.reads <= self.failures:
                raise ConnectionError("Idle socket closed")
            return b'{"state":"persisted"}'

        async def eval(self, *args):
            self.writes += 1
            raise ConnectionError("Write result unknown")

    async def exercise():
        client = Client(1)
        store = gateway.RedisPrefillStore(client)
        assert await store.get("case") == {"state": "persisted"}
        assert client.reads == 2
        client = Client(2)
        store = gateway.RedisPrefillStore(client)
        with pytest.raises(ConnectionError):
            await store.get("case")
        assert client.reads == 2
        with pytest.raises(ConnectionError):
            await store.claim("case", {"subject_hash": "u"})
        assert client.writes == 1
        with pytest.raises(ConnectionError):
            await store.update("case", "owner", {"expires_at": time.time() + 60})
        assert client.writes == 2

    asyncio.run(exercise())


def ticket(request_id, digest, size, *, operation="submit", user="u"):
    now = int(time.time())
    raw = json.dumps(
        {
            "v": 2,
            "auth_method": "api-token",
            "aud": "whitebox-prefill",
            "scope": "world",
            "sub": user,
            **WORLD,
            "request_id": request_id,
            "operation": operation,
            "payload_sha256": digest,
            "payload_bytes": size,
            "iat": now,
            "exp": now + 120,
            "jti": "a" * 32,
        },
        separators=(",", ":"),
    ).encode()
    body = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    signature = hmac.new(SECRET.encode(), body.encode(), hashlib.sha256).digest()
    return body + "." + base64.urlsafe_b64encode(signature).rstrip(b"=").decode()


class Socket:
    def __init__(self, token, raw, *, request_id="case-one", upload=True):
        self.headers = {}
        self.incoming, self.outgoing = asyncio.Queue(), asyncio.Queue()
        self.closed = False
        self.incoming.put_nowait(
            msgspec.msgpack.encode(
                {
                    "type": "prefill.begin",
                    "authorization": token,
                    "world": WORLD,
                    "request_id": request_id,
                }
            )
        )
        if upload:
            self.incoming.put_nowait(raw)

    async def accept(self):
        pass

    async def receive_bytes(self):
        return await self.incoming.get()

    async def send_bytes(self, data):
        self.outgoing.put_nowait(msgspec.msgpack.decode(data))

    async def close(self):
        self.closed = True


def test_prefill_releases_gpu_before_disk_and_reconnect_does_not_submit_twice(
    monkeypatch,
):
    async def exercise():
        disk_ready, released = asyncio.Event(), asyncio.Event()
        sent, opened = [], []

        class Upstream:
            def __init__(self):
                self.reads = 0

            async def send(self, value):
                sent.append(msgspec.msgpack.decode(value))

            async def recv(self):
                self.reads += 1
                if self.reads == 1:
                    return msgspec.msgpack.encode(
                        {"type": "prefill.ready", "capabilities": {"persistence": True}}
                    )
                value = {
                    "request_id": "case-one",
                    "session_id": "bound-session",
                    "cache_key": "b" * 64,
                }
                if self.reads == 2:
                    return msgspec.msgpack.encode(
                        {**value, "type": "prefill.memory_ready"}
                    )
                await disk_ready.wait()
                assert released.is_set()
                return msgspec.msgpack.encode(
                    {
                        **value,
                        "type": "prefill.persisted",
                        "storage": {"state": "persisted"},
                    }
                )

        class Lease:
            def __init__(self, *args, **kwargs):
                assert kwargs["operation_path"] == "/v1/realtime_video/prefill"
                assert kwargs["capacity_wait_ms"] == 0 and kwargs["strict_release"]

            @asynccontextmanager
            async def open(self):
                opened.append(True)
                yield Upstream(), NS(session_id="bound-session")

            async def release_compute(self):
                released.set()

        monkeypatch.setattr(gateway, "create_model_leases", Lease)
        store = gateway.MemoryPrefillStore()
        manager = gateway.PrefillGateway(
            coordinator=None,
            connect=None,
            secret=SECRET,
            model_revision="model",
            vae_fingerprint="local",
            store=store,
        )
        raw = msgspec.msgpack.encode({"type": "prefill", "request_id": "case-one"})
        digest = hashlib.sha256(raw).hexdigest()
        token = ticket("case-one", digest, len(raw))
        socket = Socket(token, raw)
        observer = asyncio.create_task(manager.serve(socket))
        assert (await socket.outgoing.get())["type"] == "prefill.upload_ready"
        await asyncio.wait_for(released.wait(), 2)
        key = gateway.request_key("u", WORLD, "case-one")
        assert (await store.get(key))["state"] == "memory_ready"
        disk_ready.set()
        await asyncio.wait_for(observer, 2)
        result = await store.get(key)
        assert result["state"] == "persisted" and len(opened) == 1
        assert sent[0]["persist"] is True
        other = gateway.PrefillGateway(
            coordinator=None,
            connect=None,
            secret=SECRET,
            model_revision="model",
            vae_fingerprint="local",
            store=store,
        )
        reconnect = Socket(token, raw, upload=False)
        await other.serve(reconnect)
        reply = await reconnect.outgoing.get()
        assert reply["state"] == "persisted" and len(opened) == 1
        assert "owner" not in reply and "payload_sha256" not in reply
        status_token = ticket("case-one", digest, len(raw), operation="status")
        request = NS(
            headers={"authorization": "Bearer " + status_token}, query_params=WORLD
        )
        assert (await other.status(request, "case-one"))["state"] == "persisted"
        await manager.close()
        await other.close()

    asyncio.run(exercise())


def test_prefill_upload_hash_and_audience_checked_before_claim():
    async def exercise():
        store = gateway.MemoryPrefillStore()
        manager = gateway.PrefillGateway(
            coordinator=None,
            connect=None,
            secret=SECRET,
            model_revision="model",
            vae_fingerprint="local",
            store=store,
        )
        raw = msgspec.msgpack.encode({"type": "prefill", "request_id": "case-one"})
        token = ticket("case-one", "0" * 64, len(raw))
        socket = Socket(token, raw)
        await manager.serve(socket)
        assert not store.values and not manager.tasks
        assert (await socket.outgoing.get())["type"] == "prefill.upload_ready"
        assert (await socket.outgoing.get())["code"] == "PREFILL_REQUEST_FAILED"
        rejected = Socket("invalid-token", raw)
        await manager.serve(rejected)
        assert not store.values and not manager.tasks
        assert (await rejected.outgoing.get())["code"] == "PREFILL_REQUEST_FAILED"

    asyncio.run(exercise())


def test_shared_store_owner_fences_late_updates():
    async def exercise():
        store = gateway.MemoryPrefillStore()
        first = dict(owner="one", payload_sha256="a", expires_at=time.time() + 100)
        assert (await store.claim("key", first))[0]
        assert not (await store.claim("key", {**first, "owner": "two"}))[0]
        try:
            await store.update("key", "two", {**first, "state": "persisted"})
        except RuntimeError:
            pass
        else:
            raise AssertionError("Late owner update accepted")
        assert "state" not in await store.get("key")

    asyncio.run(exercise())


def test_actual_lease_releases_once_while_disk_socket_stays_open():
    from urllib.parse import parse_qs, urlsplit

    from sglang.multimodal_gen.runtime.realtime.whitebox.interactive import ModelLeases
    from sglang.multimodal_gen.test.unit.realtime.test_whitebox_strict_release import (
        setup,
    )

    async def exercise():
        coordinator, store, reservations, previous, _ = await setup()
        await coordinator.release_verified(previous)

        class Upstream:
            closed = False

            async def close(self):
                self.closed = True

        upstream = Upstream()

        async def connect(url, **kwargs):
            parts = urlsplit(url)
            assert parts.path == "/v1/realtime_video/prefill"
            assert parse_qs(parts.query).get("session_id")
            return upstream

        class Client:
            async def admit(self, **kwargs):
                kwargs.pop("trace_id", None)
                return await coordinator.admit(**kwargs)

            async def release_verified(self, value):
                return await coordinator.release_verified(value)

            async def renew(self, value):
                return await coordinator.renew(value)

            async def quarantine(self, value):
                return await coordinator.quarantine(value)

        lease = ModelLeases(
            Client(),
            connect,
            user_id="prefill-user",
            model_revision="minwm-r1",
            vae_fingerprint="local",
            trace_id="trace",
            max_bytes=1024,
            strict_release=True,
            operation_path="/v1/realtime_video/prefill",
        )
        async with lease.open():
            assert (await store.capacity_snapshot())["roles"]["denoiser"][
                "free_slots"
            ] == 0
            await lease.release_compute()
            assert not upstream.closed
            assert (await store.capacity_snapshot())["roles"]["denoiser"][
                "free_slots"
            ] == 1
            count = reservations.releases
            await lease.release_compute()
            assert reservations.releases == count
        assert upstream.closed and reservations.releases == count

    asyncio.run(exercise())
